#!/usr/bin/env python3
"""Graft NVFP4 MTP draft experts onto a compressed-tensors checkpoint.

The port of scripts/prepare-mtp-graft.sh to the MODE=ct family. Same idea -- the
MTP draft head ships with its routed experts as fused BF16 (~5 GB), and Inferact
publishes the same experts in NVFP4 (~1.4 GB), so swapping them frees ~3.4 GiB --
but two things differ from the ModelOpt case and both matter.

**The donor has to be transcoded.** Inferact's shard is ModelOpt-format; the target
is compressed-tensors. The tensors are the same numbers under different names, plus
one convention difference:

    ModelOpt                     compressed-tensors
    weight            (U8)   ->  weight_packed        (U8)   verbatim
    weight_scale  (F8_E4M3)  ->  weight_scale     (F8_E4M3)  verbatim
    weight_scale_2    (F32)  ->  weight_global_scale  (F32)  RECIPROCAL
    input_scale       (F32)  ->  dropped

The reciprocal is not a guess. vLLM feeds both formats into the same
`convert_to_nvfp4_moe_kernel_format`, and the two call sites differ exactly there:
ModelOpt passes `w13_scale_2=layer.w13_weight_scale_2` while compressed-tensors
passes `w13_scale_2=(1.0 / layer.w13_weight_global_scale)`. Checked against the
main-model experts of two independent quantizations of this base model, the rule
reproduces `down_proj` global scales to 0.3%.

`input_scale` is dropped because these checkpoints carry no activation scales; the
MTP experts then read as weight-only NVFP4, like the main model's, and land on the
same kernel.

**The config surgery is smaller.** ModelOpt's exclusion is a blanket `mtp.*` glob
that also covers the draft's attention and shared expert, so prepare-mtp-graft.sh
has to replace it with 29 explicit module names. compressed-tensors targets its
routed experts with `re:.*mlp\\.experts\\..*proj$`, which matches the MTP experts
and nothing else under `mtp.` -- the draft's `self_attn.q_proj`,
`mlp.shared_expert.gate_proj`, `fc_embedding` and friends contain no
`mlp.experts.` segment. So dropping the `re:.*mtp\\..*` ignore entry is sufficient
and safe: it quantizes exactly the routed experts and leaves the rest BF16.

Everything streams. The donor is ~1.4 GB and the MTP shard ~5 GB, and this may run
beside a live server on a box with little free memory, so tensors are copied
through a small buffer rather than materialised.

Usage:  ct_mtp_graft.py <ctprep-snapshot> --donor <shard.safetensors> [--out <dir>]
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import struct
import sys

EXPERT_RE = re.compile(r"^mtp\.layers\.\d+\.mlp\.experts\.\d+\.(gate|up|down)_proj\.(\w+)$")
FUSED = ("mtp.layers.0.mlp.experts.down_proj", "mtp.layers.0.mlp.experts.gate_up_proj")
MTP_IGNORE_PATTERNS = ("re:.*mtp\\..*", "re:.*mtp\\.*", "mtp.*", "model.mtp.*")
EXPERT_TENSORS = 6144          # 512 experts x 3 projections x 4 tensors
SHARED_TENSORS = 29            # the draft's non-expert tensors, left in the target
OUT_SHARD = "mtp_experts_nvfp4.safetensors"
COPY_BUF = 1 << 22


def die(msg: str) -> None:
    sys.exit(f"!! {msg}")


def read_header(path: str) -> tuple[dict, int]:
    with open(path, "rb") as f:
        (n,) = struct.unpack("<Q", f.read(8))
        header = json.loads(f.read(n))
    header.pop("__metadata__", None)
    return header, 8 + n


def write_safetensors(out_path: str, entries: list[tuple[str, dict, callable]]) -> None:
    """entries: (name, {dtype, shape}, writer(fh) -> bytes_written), in order."""
    header, offset = {}, 0
    for name, meta, _ in entries:
        nbytes = meta["nbytes"]
        header[name] = {
            "dtype": meta["dtype"],
            "shape": meta["shape"],
            "data_offsets": [offset, offset + nbytes],
        }
        offset += nbytes
    blob = json.dumps(header, separators=(",", ":")).encode()
    pad = (-(8 + len(blob))) % 8            # data must start 8-byte aligned
    blob += b" " * pad
    tmp = out_path + ".incomplete"
    with open(tmp, "wb") as fh:
        fh.write(struct.pack("<Q", len(blob)))
        fh.write(blob)
        written = 0
        for name, meta, writer in entries:
            n = writer(fh)
            if n != meta["nbytes"]:
                die(f"{name}: wrote {n} bytes, header says {meta['nbytes']}")
            written += n
    if written != offset:
        die(f"payload {written} != expected {offset}")
    os.replace(tmp, out_path)


def copy_range(src: str, start: int, nbytes: int):
    def writer(fh) -> int:
        done = 0
        with open(src, "rb") as sf:
            sf.seek(start)
            while done < nbytes:
                chunk = sf.read(min(COPY_BUF, nbytes - done))
                if not chunk:
                    break
                fh.write(chunk)
                done += len(chunk)
        return done

    return writer


def reciprocal_f32(src: str, start: int, nbytes: int):
    """weight_scale_2 -> weight_global_scale. Scalar or tiny; read it whole."""
    def writer(fh) -> int:
        with open(src, "rb") as sf:
            sf.seek(start)
            raw = sf.read(nbytes)
        vals = struct.unpack(f"<{nbytes // 4}f", raw)
        for v in vals:
            if v == 0.0:
                die("weight_scale_2 of 0 cannot be inverted")
        fh.write(struct.pack(f"<{len(vals)}f", *(1.0 / v for v in vals)))
        return nbytes

    return writer


def transcode_donor(donor: str, out_path: str) -> list[str]:
    header, data_start = read_header(donor)
    experts = {k: v for k, v in header.items() if EXPERT_RE.match(k)}
    shared = {k: v for k, v in header.items() if k not in experts}
    stray = [k for k in header if not k.startswith("mtp.")]
    if stray:
        die(f"donor holds {len(stray)} tensors outside mtp. (e.g. {stray[0]}) -- refusing")
    if any(k in header for k in FUSED):
        die("donor unexpectedly contains fused expert tensors")
    if len(experts) != EXPERT_TENSORS or len(shared) != SHARED_TENSORS:
        die(f"unexpected donor layout: {len(experts)} expert / {len(shared)} shared "
            f"(want {EXPERT_TENSORS}/{SHARED_TENSORS})")
    print(f">> donor: {len(experts)} expert tensors + {len(shared)} shared (left alone)")

    rename = {"weight": "weight_packed", "weight_scale_2": "weight_global_scale"}
    entries, names = [], []
    for name in sorted(experts):
        m = EXPERT_RE.match(name)
        suffix = m.group(2)
        if suffix == "input_scale":
            continue                        # weight-only: no activation scales
        meta = experts[name]
        start, end = meta["data_offsets"]
        nbytes = end - start
        out_name = name.rsplit(".", 1)[0] + "." + rename.get(suffix, suffix)
        spec = {"dtype": meta["dtype"], "shape": meta["shape"], "nbytes": nbytes}
        if suffix == "weight_scale_2":
            if meta["dtype"] != "F32":
                die(f"{name}: expected F32 weight_scale_2, got {meta['dtype']}")
            writer = reciprocal_f32(donor, data_start + start, nbytes)
        else:
            writer = copy_range(donor, data_start + start, nbytes)
        entries.append((out_name, spec, writer))
        names.append(out_name)
    print(f">> transcoding {len(entries)} tensors -> {os.path.basename(out_path)}")
    write_safetensors(out_path, entries)
    print(f">> wrote {os.path.getsize(out_path) / 1e9:.2f} GB")
    return names


def rewrite_without(src: str, out_path: str, drop: set[str]) -> None:
    """Copy a shard, omitting `drop`. The loader walks whole files, so removing the
    keys from the index alone would still hand the fused BF16 experts to the model."""
    header, data_start = read_header(src)
    missing = drop - set(header)
    if missing:
        die(f"{os.path.basename(src)} does not contain {sorted(missing)}")
    entries = []
    for name in sorted(header, key=lambda k: header[k]["data_offsets"][0]):
        if name in drop:
            continue
        meta = header[name]
        start, end = meta["data_offsets"]
        entries.append((
            name,
            {"dtype": meta["dtype"], "shape": meta["shape"], "nbytes": end - start},
            copy_range(src, data_start + start, end - start),
        ))
    write_safetensors(out_path, entries)
    print(f">> rewrote {os.path.basename(out_path)}: {len(header)} -> {len(entries)} tensors "
          f"({os.path.getsize(out_path) / 1e9:.2f} GB)")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("target", help="the -ctprep snapshot to graft onto")
    ap.add_argument("--donor", required=True, help="Inferact nvfp4_experts_mtp.safetensors")
    ap.add_argument("--out", default="", help="default: <target>-mtpnvfp4")
    args = ap.parse_args()

    target = args.target.rstrip("/")
    dst = args.out or target + "-mtpnvfp4"
    if not os.path.isdir(target):
        die(f"{target} is not a directory")
    if not os.path.exists(args.donor):
        die(f"donor {args.donor} not found")
    if os.path.exists(os.path.join(dst, ".prepared")):
        print(f">> already prepared: {dst}")
        return

    if os.path.exists(dst):
        shutil.rmtree(dst)
    # copytree with symlinks=True keeps the relative ../../blobs/<sha> links as links.
    shutil.copytree(target, dst, symlinks=True)
    print(f">> {dst}")

    index_path = os.path.join(dst, "model.safetensors.index.json")
    index = json.load(open(index_path))
    weight_map = index["weight_map"]

    fused_files = {weight_map[k] for k in FUSED if k in weight_map}
    if len(fused_files) != 1:
        die(f"expected the fused MTP experts in one file, found {sorted(fused_files)}")
    mtp_file = fused_files.pop()
    others = [k for k, v in weight_map.items() if v == mtp_file and k not in FUSED]
    if any(not k.startswith("mtp.") for k in others):
        die(f"{mtp_file} holds non-MTP tensors; rewriting it would disturb them")
    print(f">> fused BF16 experts live in {mtp_file} ({len(others)} other MTP tensors alongside)")

    # 1. donor -> compressed-tensors naming
    new_names = transcode_donor(args.donor, os.path.join(dst, OUT_SHARD))

    # 2. the MTP shard without the two fused tensors
    src_shard = os.path.realpath(os.path.join(dst, mtp_file))
    tmp_shard = os.path.join(dst, mtp_file + ".new")
    rewrite_without(src_shard, tmp_shard, set(FUSED))
    os.replace(tmp_shard, os.path.join(dst, mtp_file))   # replaces the symlink

    # 3. index
    for k in FUSED:
        weight_map.pop(k, None)
    for name in new_names:
        weight_map[name] = OUT_SHARD
    with open(index_path, "w") as f:
        json.dump(index, f)
    print(f">> index: -{len(FUSED)} fused, +{len(new_names)} NVFP4 -> {len(weight_map)} tensors")

    # 4. let the quantization config see the MTP experts
    cfg_path = os.path.join(dst, "config.json")
    cfg = json.load(open(cfg_path))
    quant = cfg.get("quantization_config") or {}
    ignore = quant.get("ignore")
    if ignore is None:
        die("config.json has no quantization_config.ignore")
    removed = [p for p in ignore if p in MTP_IGNORE_PATTERNS]
    if not removed:
        die(f"no MTP ignore pattern found; expected one of {MTP_IGNORE_PATTERNS}")
    quant["ignore"] = [p for p in ignore if p not in MTP_IGNORE_PATTERNS]
    # Anything else under mtp. must stay unquantized. Only the routed experts carry
    # an `mlp.experts.` segment, so nothing else can match the expert target, but
    # assert it rather than trust it.
    targets = [t for g in quant.get("config_groups", {}).values() for t in g.get("targets", [])]
    expert_pats = [t[3:] for t in targets if t.startswith("re:")]
    for name in ("mtp.layers.0.self_attn.q_proj", "mtp.layers.0.mlp.shared_expert.gate_proj",
                 "mtp.fc_embedding", "mtp.fc_hidden", "mtp.layers.0.mlp.gate"):
        for pat in expert_pats:
            if re.search(pat, name):
                die(f"removing the MTP ignore would also quantize {name} via {pat!r}")
    with open(cfg_path, "w") as f:
        json.dump(cfg, f, indent=2)
    print(f">> config.json: dropped {removed} from quantization ignore")
    print("   (only mtp routed experts match the expert target; the rest stays BF16)")

    # 5. everything the index names must resolve
    missing = sorted({fn for fn in set(weight_map.values())
                      if not os.path.exists(os.path.join(dst, fn))})
    if missing:
        die(f"index references missing files: {missing}")
    open(os.path.join(dst, ".prepared"), "w").close()
    print(f">> done: {len(set(weight_map.values()))} files, {len(weight_map)} tensors")


if __name__ == "__main__":
    main()
