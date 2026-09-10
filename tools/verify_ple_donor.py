#!/usr/bin/env python3
"""Check that a donor FP8 PLE table really is the same table as the target's.

``tools/ct_prepare.py --ple donor`` substitutes an FP8 n-gram table from one
checkpoint into another. That is only sound if both descend from the same base
model *and* the derivative did not touch the table. The n-gram table is a pure
lookup -- abliterations and most fine-tunes edit projections, not embeddings --
but "almost certainly" is not a reason to serve 47.7 GiB of someone else's
weights without looking.

This samples rows from the target's own table and compares them against the
dequantized donor. The target's table does not have to be on disk: with
``--repo`` the rows are pulled with HTTP range requests, which is a few hundred
KiB rather than the ~102 GB the shard weighs.

What "the same table" looks like: the donor is an FP8-e4m3 quantization of the
same BF16 values, so the two agree to FP8 rounding. e4m3 has a 3-bit mantissa,
so the relative step is up to 1/16 = 6.25%; a p99 relative error comfortably
under that, with correlation ~0.9996, means same table. A different table
diverges immediately and obviously -- correlation drops toward zero.

Usage:
  verify_ple_donor.py --donor <snapshot> --repo <org/name> [--revision main]
  verify_ple_donor.py --donor <snapshot> --target <snapshot>
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import random
import re
import struct
import sys
import urllib.request

import numpy as np
import torch

SHARD_RE = re.compile(
    r"layers\.(\d+)\.ple\.ple_embedding\.ngram_embedding\.shard_(\d+)\.weight$"
)
SCALE_RE = re.compile(
    r"layers\.(\d+)\.ple\.ple_embedding\.ngram_embedding\.weight_scale$"
)
# e4m3 has a 3-bit mantissa: the worst-case relative step between representable
# values is 2^-4 = 6.25%. Leave a little headroom for the tail.
FP8_P99_CEILING = 0.07


def die(msg: str) -> None:
    sys.exit(f"!! {msg}")


def read_header(path: str) -> tuple[dict, int]:
    with open(path, "rb") as f:
        (n,) = struct.unpack("<Q", f.read(8))
        header = json.loads(f.read(n))
    header.pop("__metadata__", None)
    return header, 8 + n


def bf16_to_f32(raw: bytes) -> np.ndarray:
    return (np.frombuffer(raw, dtype=np.uint16).astype(np.uint32) << 16).view(np.float32)


def load_donor(donor: str) -> tuple[dict[int, tuple[str, int]], float, int]:
    """Return ({shard: (path, data_offset)}, global_scale, cols)."""
    shards: dict[int, tuple[str, int]] = {}
    scale = None
    cols = 0
    for path in sorted(glob.glob(os.path.join(donor, "*.safetensors"))):
        header, data_start = read_header(path)
        for name, meta in header.items():
            if m := SHARD_RE.search(name):
                if meta["dtype"] not in ("F8_E4M3", "F8_E5M2"):
                    die(f"donor shard {name} is {meta['dtype']}, not FP8")
                shards[int(m.group(2))] = (path, data_start + meta["data_offsets"][0])
                cols = meta["shape"][1]
            elif SCALE_RE.search(name):
                with open(path, "rb") as f:
                    f.seek(data_start + meta["data_offsets"][0])
                    raw = f.read(meta["data_offsets"][1] - meta["data_offsets"][0])
                scale = float(bf16_to_f32(raw)[0]) if meta["dtype"] == "BF16" else float(
                    np.frombuffer(raw, dtype=np.float32)[0]
                )
    if not shards:
        die(f"no PLE shards under {donor}")
    if scale is None:
        die("donor has no ngram_embedding.weight_scale")
    return shards, scale, cols


class TargetTable:
    """Row reader for the target's table, local file or HTTP range requests."""

    def __init__(self, header: dict, data_start: int, reader) -> None:
        self.header, self.data_start, self._read = header, data_start, reader

    def rows(self, layer: int, shard: int, row0: int, n: int, cols: int) -> np.ndarray:
        key = (
            f"model.language_model.layers.{layer}"
            f".ple.ple_embedding.ngram_embedding.shard_{shard}.weight"
        )
        meta = self.header[key]
        if meta["dtype"] != "BF16":
            die(f"target shard is {meta['dtype']}; this check assumes a BF16 table")
        start = self.data_start + meta["data_offsets"][0] + row0 * cols * 2
        return bf16_to_f32(self._read(start, n * cols * 2)).reshape(n, cols)

    def shard_rows(self, layer: int, shard: int) -> int:
        key = (
            f"model.language_model.layers.{layer}"
            f".ple.ple_embedding.ngram_embedding.shard_{shard}.weight"
        )
        return self.header[key]["shape"][0]


def open_remote(repo: str, revision: str, token: str) -> TargetTable:
    base = f"https://huggingface.co/{repo}/resolve/{revision}"
    index_url = f"{base}/model.safetensors.index.json"
    headers = {"Authorization": f"Bearer {token}"} if token else {}

    def fetch(url: str, byte_range: tuple[int, int] | None = None) -> bytes:
        h = dict(headers)
        if byte_range:
            h["Range"] = f"bytes={byte_range[0]}-{byte_range[1]}"
        return urllib.request.urlopen(
            urllib.request.Request(url, headers=h), timeout=300
        ).read()

    weight_map = json.loads(fetch(index_url))["weight_map"]
    files = {fn for name, fn in weight_map.items() if SHARD_RE.search(name)}
    if len(files) != 1:
        die(f"expected the PLE shards in one file, found {sorted(files)}")
    shard_url = f"{base}/{files.pop()}"
    n = struct.unpack("<Q", fetch(shard_url, (0, 7)))[0]
    header = json.loads(fetch(shard_url, (8, 8 + n - 1)))
    header.pop("__metadata__", None)
    return TargetTable(header, 8 + n, lambda a, ln: fetch(shard_url, (a, a + ln - 1)))


def open_local(snapshot: str) -> TargetTable:
    weight_map = json.load(open(os.path.join(snapshot, "model.safetensors.index.json")))
    files = {fn for name, fn in weight_map["weight_map"].items() if SHARD_RE.search(name)}
    if len(files) != 1:
        die(f"expected the PLE shards in one file, found {sorted(files)}")
    path = os.path.join(snapshot, files.pop())
    header, data_start = read_header(path)

    def read(a: int, ln: int) -> bytes:
        with open(path, "rb") as f:
            f.seek(a)
            return f.read(ln)

    return TargetTable(header, data_start, read)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--donor", required=True, help="snapshot with the FP8 table")
    ap.add_argument("--repo", default="", help="HF repo id of the target (range reads)")
    ap.add_argument("--revision", default="main")
    ap.add_argument("--target", default="", help="local target snapshot instead of --repo")
    ap.add_argument("--samples", type=int, default=10, help="blocks to compare")
    ap.add_argument("--rows", type=int, default=256, help="rows per block")
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    if bool(args.repo) == bool(args.target):
        die("pass exactly one of --repo or --target")

    donor_shards, scale, cols = load_donor(args.donor)
    print(f">> donor: {len(donor_shards)} FP8 shards, {cols} cols, global scale {scale:.10g}")

    token = os.environ.get("HF_TOKEN", "")
    target = open_remote(args.repo, args.revision, token) if args.repo else open_local(args.target)
    layer = {int(m.group(1)) for k in target.header if (m := SHARD_RE.search(k))}.pop()

    random.seed(args.seed)
    rels, worst_abs, sampled = [], 0.0, 0
    for _ in range(args.samples):
        shard = random.choice(sorted(donor_shards))
        total = target.shard_rows(layer, shard)
        row0 = random.randrange(0, max(1, total - args.rows))
        ref = target.rows(layer, shard, row0, args.rows, cols)

        path, offset = donor_shards[shard]
        with open(path, "rb") as f:
            f.seek(offset + row0 * cols)
            raw = f.read(args.rows * cols)
        got = (
            torch.frombuffer(bytearray(raw), dtype=torch.float8_e4m3fn)
            .float()
            .numpy()
            .reshape(args.rows, cols)
            * scale
        )
        diff = np.abs(ref - got)
        rel = diff / np.maximum(np.abs(ref), 1e-12)
        corr = float(np.corrcoef(ref.ravel(), got.ravel())[0, 1])
        rels.append(rel.ravel())
        worst_abs = max(worst_abs, float(diff.max()))
        sampled += args.rows
        print(
            f"   shard {shard:3d} rows {row0}: max|d|={diff.max():.6f} "
            f"p99rel={np.percentile(rel, 99):.4f} corr={corr:.6f}"
        )

    rel = np.concatenate(rels)
    p99 = float(np.percentile(rel, 99))
    print(f"\n>> {sampled} rows / {rel.size} values sampled")
    print(f"   p50 rel {np.percentile(rel, 50):.5f} | p99 rel {p99:.5f} | max abs {worst_abs:.6f}")
    if p99 > FP8_P99_CEILING:
        die(
            f"p99 relative error {p99:.4f} exceeds the {FP8_P99_CEILING} FP8 rounding "
            "ceiling -- these are NOT the same table; do not splice"
        )
    print(f">> within FP8-e4m3 rounding ({FP8_P99_CEILING}): same table, safe to splice")


if __name__ == "__main__":
    main()
