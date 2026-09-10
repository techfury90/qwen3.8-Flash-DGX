#!/usr/bin/env python3
"""Prepare a compressed-tensors Qwen3.8-Flash-Next checkpoint for this recipe.

Two checkpoint families exist for this model. The one the repo is built on
(RadixArk) is ModelOpt-NVFP4: routed experts in NVFP4, everything else excluded
and left BF16, and the 51.2B-parameter n-gram (PLE) table stored in FP8-e4m3
with one global scale. The other family is llm-compressor / compressed-tensors
output, which makes the opposite choices: the side layers arrive already 8-bit,
and the PLE table is left in BF16 -- 95.4 GiB instead of 47.7 GiB.

This script builds a sibling snapshot that the recipe can serve, doing only what
is actually necessary:

  1. ``layer_types``. transformers >= 5.16 renamed the sparse-attention layer
     type from ``full_attention`` to ``qwen_sparse_attention``. The vLLM in this
     image knows only the old name: ``Qwen3_8FlashNextDecoderLayer.__init__``
     raises ``Invalid layer_type``, and ``_qsa_layer_ids`` (which drives the QSA
     cache-scale remapping) silently comes out empty. Both are fixed by renaming
     the 12 entries back. ``full_attention`` + ``indexer_n_heads`` is exactly how
     the old naming spelled "QSA layer", so this is a rename, not a behaviour
     change.

  2. The PLE table (``PLE=donor``, the default). A BF16 table costs 5,120 bytes
     of NVMe per token instead of 2,560, and halves how much of it the page cache
     can hold next to the weights. The table is a pure lookup that fine-tunes and
     abliterations do not touch, so an FP8 table from a donor checkpoint of the
     same base model can be substituted wholesale: the 128 shard tensors have the
     same names, the same geometry and the same row ordering in both. Only the
     index is rewritten -- the donor's shard files are symlinked in, nothing is
     copied and neither parent snapshot is touched.

     Verify the substitution first with ``tools/verify_ple_donor.py``, which
     samples rows from the target's own table over HTTP range requests and
     compares them against the dequantized donor. ``PLE=keep`` skips the splice
     and serves the checkpoint's own BF16 table (supported, just slower).

Everything else is left exactly as published. In particular the quantization
config is NOT rewritten: with no activation scales in the checkpoint, vLLM reads
the experts as weight-only NVFP4 and the MoE runs on Marlin rather than
FLASHINFER_CUTLASS. That is a real difference in speed profile and costs
deterministic greedy decoding; see the README.

Usage:  ct_prepare.py <target-snapshot> [--donor <donor-snapshot>] [--ple donor|keep]
"""

from __future__ import annotations

import argparse
import json
import os
import re
import struct
import sys

SHARD_RE = re.compile(
    r"layers\.(\d+)\.ple\.ple_embedding\.ngram_embedding\.shard_(\d+)\.weight$"
)
SCALE_RE = re.compile(
    r"layers\.(\d+)\.ple\.ple_embedding\.ngram_embedding\.weight_scale$"
)
# The PLE geometry has to agree between donor and target or the rows do not line
# up. These are the config keys the table's shape and row ordering derive from.
GEOMETRY_KEYS = (
    "split_ngram_parts",
    "ngram_vocab_size_base",
    "heads_per_ngram",
    "make_ngram_vocab_size_divisible_by",
    "ple_embed_dim",
    "ple_layer_ids",
    "ngram_size",
)
FP8_DTYPES = {"F8_E4M3", "F8_E5M2"}


def die(msg: str) -> None:
    sys.exit(f"!! {msg}")


def read_header(path: str) -> tuple[dict, int]:
    """Return (header, data_start) of a safetensors file."""
    with open(path, "rb") as f:
        (n,) = struct.unpack("<Q", f.read(8))
        header = json.loads(f.read(n))
    header.pop("__metadata__", None)
    return header, 8 + n


def load_index(snapshot: str) -> dict:
    path = os.path.join(snapshot, "model.safetensors.index.json")
    if not os.path.exists(path):
        die(f"no model.safetensors.index.json under {snapshot}")
    with open(path) as f:
        return json.load(f)


def fix_layer_types(config: dict) -> int:
    """Rename qwen_sparse_attention -> full_attention. Returns the count."""
    text = config.get("text_config")
    if text is None:
        die("config.json has no text_config")
    types = text.get("layer_types")
    if not types:
        die("config.json text_config has no layer_types")
    renamed = sum(1 for t in types if t == "qwen_sparse_attention")
    text["layer_types"] = [
        "full_attention" if t == "qwen_sparse_attention" else t for t in types
    ]
    unknown = {t for t in text["layer_types"]} - {"full_attention", "linear_attention"}
    if unknown:
        die(f"layer_types holds names this vLLM does not know: {sorted(unknown)}")
    if text["layer_types"].count("full_attention") and not text.get("indexer_n_heads"):
        die("full_attention layers without indexer_n_heads: this is not a QSA model")
    return renamed


def fix_vision_model_type(config: dict) -> bool:
    """qwen4_exp_vision -> qwen4_exp, matching what Qwen4ExpVisionConfig declares."""
    vision = config.get("vision_config")
    if isinstance(vision, dict) and vision.get("model_type") == "qwen4_exp_vision":
        vision["model_type"] = "qwen4_exp"
        return True
    return False


def ple_layer_index(weight_map: dict) -> int:
    layers = {int(m.group(1)) for k in weight_map if (m := SHARD_RE.search(k))}
    if len(layers) != 1:
        die(f"expected PLE shards under exactly one layer, found {sorted(layers)}")
    return layers.pop()


def check_geometry(target_cfg: dict, donor_cfg: dict) -> None:
    t, d = target_cfg.get("text_config", {}), donor_cfg.get("text_config", {})
    for key in GEOMETRY_KEYS:
        if t.get(key) != d.get(key):
            die(
                f"PLE geometry differs on {key!r}: target={t.get(key)!r} "
                f"donor={d.get(key)!r} -- the tables are not interchangeable"
            )


def collect_donor_shards(donor: str, layer_idx: int) -> tuple[dict[int, str], str, str]:
    """Return ({shard_idx: filename}, scale_filename, dtype) from the donor snapshot."""
    index = load_index(donor)["weight_map"]
    shards: dict[int, str] = {}
    scale_file = ""
    for name, fname in index.items():
        if (m := SHARD_RE.search(name)) and int(m.group(1)) == layer_idx:
            shards[int(m.group(2))] = fname
        elif (m := SCALE_RE.search(name)) and int(m.group(1)) == layer_idx:
            scale_file = fname
    if not shards:
        die(f"donor has no PLE shards for layer {layer_idx}")

    dtype = ""
    cols = None
    for idx, fname in sorted(shards.items()):
        header, _ = read_header(os.path.join(donor, fname))
        key = next(k for k in header if SHARD_RE.search(k) and k.endswith(f".shard_{idx}.weight"))
        meta = header[key]
        if dtype and meta["dtype"] != dtype:
            die("donor PLE shards have mixed dtypes")
        dtype = meta["dtype"]
        if cols is not None and meta["shape"][1] != cols:
            die("donor PLE shards have mixed row widths")
        cols = meta["shape"][1]
    if dtype in FP8_DTYPES and not scale_file:
        die("donor PLE table is FP8 but carries no ngram_embedding.weight_scale")
    return shards, scale_file, dtype


def relative_link(dst_dir: str, target_path: str) -> str:
    return os.path.relpath(os.path.realpath(target_path), dst_dir)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("target", help="snapshot directory to prepare (modified in place)")
    ap.add_argument("--donor", default="", help="snapshot holding an FP8 PLE table")
    ap.add_argument("--ple", choices=("donor", "keep"), default="donor")
    args = ap.parse_args()

    dst = args.target.rstrip("/")
    if not os.path.isdir(dst):
        die(f"{dst} is not a directory")

    # --- 1. config.json -----------------------------------------------------
    cfg_path = os.path.join(dst, "config.json")
    with open(cfg_path) as f:
        config = json.load(f)
    renamed = fix_layer_types(config)
    vision_fixed = fix_vision_model_type(config)
    print(f">> layer_types: renamed {renamed} qwen_sparse_attention -> full_attention")
    if vision_fixed:
        print(">> vision_config.model_type: qwen4_exp_vision -> qwen4_exp")

    index = load_index(dst)
    weight_map = index["weight_map"]
    layer_idx = ple_layer_index(weight_map)
    target_shards = {
        int(m.group(2)): fname
        for name, fname in weight_map.items()
        if (m := SHARD_RE.search(name))
    }
    print(f">> PLE: {len(target_shards)} shard tensors under layers.{layer_idx}")

    # --- 2. the PLE table ---------------------------------------------------
    if args.ple == "donor":
        if not args.donor:
            die("--ple donor needs --donor <snapshot>")
        if not os.path.isdir(args.donor):
            die(f"donor snapshot {args.donor} not found -- download it first")
        with open(os.path.join(args.donor, "config.json")) as f:
            donor_cfg = json.load(f)
        check_geometry(config, donor_cfg)
        print(">> PLE geometry matches the donor on every shape-determining key")

        donor_shards, scale_file, dtype = collect_donor_shards(args.donor, layer_idx)
        if set(donor_shards) != set(target_shards):
            die(
                f"donor has shards {min(donor_shards)}..{max(donor_shards)} "
                f"({len(donor_shards)}), target needs {len(target_shards)}"
            )
        print(f">> donor PLE table: {len(donor_shards)} shards, dtype {dtype}")

        # Symlink the donor's shard files in, then repoint the index at them.
        linked = set()
        for fname in set(donor_shards.values()) | ({scale_file} if scale_file else set()):
            src = os.path.join(args.donor, fname)
            link = os.path.join(dst, fname)
            if os.path.lexists(link):
                os.remove(link)
            os.symlink(relative_link(dst, src), link)
            linked.add(fname)
        print(f">> symlinked {len(linked)} donor shard files into the snapshot")

        stale = set()
        for name, fname in list(weight_map.items()):
            if (m := SHARD_RE.search(name)) and int(m.group(1)) == layer_idx:
                stale.add(fname)
                weight_map[name] = donor_shards[int(m.group(2))]
        if scale_file:
            scale_name = (
                f"model.language_model.layers.{layer_idx}"
                ".ple.ple_embedding.ngram_embedding.weight_scale"
            )
            weight_map[scale_name] = scale_file
            print(f">> added {scale_name.rsplit('.', 1)[-1]} to the index")

        # The file that used to hold the BF16 table is now referenced by nothing.
        for fname in stale:
            if fname in weight_map.values():
                continue
            path = os.path.join(dst, fname)
            if os.path.lexists(path):
                os.remove(path)
            print(f">> dropped {fname} (no longer referenced)")
    else:
        print(">> PLE: keeping the checkpoint's own table (BF16 tables are supported)")

    # --- 3. write it back ---------------------------------------------------
    with open(cfg_path, "w") as f:
        json.dump(config, f, indent=2)
    with open(os.path.join(dst, "model.safetensors.index.json"), "w") as f:
        json.dump(index, f)

    # Every file the index names must exist and be readable.
    missing = sorted(
        {fn for fn in set(weight_map.values()) if not os.path.exists(os.path.join(dst, fn))}
    )
    if missing:
        die(f"index references {len(missing)} missing file(s), e.g. {missing[:3]}")
    print(f">> index resolves: {len(set(weight_map.values()))} files, {len(weight_map)} tensors")


if __name__ == "__main__":
    main()
