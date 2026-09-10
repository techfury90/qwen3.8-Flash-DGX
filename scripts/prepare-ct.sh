#!/usr/bin/env bash
# One-time preparation of a compressed-tensors Qwen3.8-Flash-Next checkpoint
# (llm-compressor output) so this recipe can serve it.
#
#   MODEL=orcarouter/Qwen3.8-Flash-Next-Uncensored-NVFP4 scripts/prepare-ct.sh
#   MODEL=orcarouter/Qwen3.8-Flash-Next-Uncensored-NVFP4 MODE=ct scripts/serve.sh
#
# Two things differ from the ModelOpt checkpoints the repo is built on, and this
# script fixes exactly those:
#
#   - transformers >= 5.16 renamed the sparse-attention layer type to
#     "qwen_sparse_attention". The vLLM in this image knows only the old
#     "full_attention" spelling and raises "Invalid layer_type" on it (and would
#     silently build an empty QSA layer set even if it did not). Renamed back.
#
#   - The n-gram (PLE) table is left in BF16: 95.4 GiB on disk instead of the
#     47.7 GiB an FP8 table takes, which doubles the NVMe read per token and
#     halves what the page cache can hold. The table is a pure lookup that
#     abliterations and fine-tunes do not touch, so the FP8 table from a donor
#     checkpoint of the same base model is substituted in by symlink + an index
#     rewrite. Set PLE=keep to serve the checkpoint's own BF16 table instead
#     (supported, just slower).
#
# VERIFY THE DONOR FIRST. tools/verify_ple_donor.py samples rows from the
# target's own table -- over HTTP range requests, so the 102 GB shard need not be
# downloaded -- and compares them against the dequantized donor:
#
#   HF_TOKEN=$(cat ~/.cache/huggingface/token) docker run --rm \
#     -e HF_TOKEN -v "$HOME/.cache/huggingface:/hf:ro" -v "$PWD/tools:/tools:ro" \
#     --entrypoint python3 qwen38-flash-dgx /tools/verify_ple_donor.py \
#     --donor /hf/hub/models--RadixArk--Qwen3.8-Flash-Next-NVFP4/snapshots/<rev> \
#     --repo orcarouter/Qwen3.8-Flash-Next-Uncensored-NVFP4
#
# What this does NOT do: rewrite the quantization config. These checkpoints carry
# no activation scales, so vLLM reads the experts as weight-only NVFP4 and the
# MoE runs on Marlin instead of FLASHINFER_CUTLASS -- faster raw decode, slower
# prefill, and no deterministic greedy decoding. See the README.
#
# Layout: a sibling of the HF snapshot, <snapshot>-ctprep/, made of the same
# relative symlinks into blobs/ -- only config.json and the index are real files.
# Nothing in either parent snapshot is touched.
set -euo pipefail

MODEL="${MODEL:?set MODEL to the compressed-tensors repo, e.g. org/Name}"
PLE_DONOR="${PLE_DONOR:-RadixArk/Qwen3.8-Flash-Next-NVFP4}"
PLE="${PLE:-donor}"
IMAGE="${IMAGE:-qwen38-flash-dgx}"
HF_CACHE="${HF_CACHE:-$HOME/.cache/huggingface}"

# Same revision resolution as scripts/serve.sh - the two must agree on the snapshot.
resolve_snapshot() {  # $1 = repo id, $2 = suffix to exclude
  local repo_dir="$HF_CACHE/hub/models--${1//\//--}" ref rev out=""
  for ref in main master; do
    rev="$(cat "$repo_dir/refs/$ref" 2>/dev/null || true)"
    if [ -n "$rev" ] && [ -d "$repo_dir/snapshots/$rev" ]; then out="$repo_dir/snapshots/$rev/"; break; fi
  done
  [ -n "$out" ] || out="$(ls -dt "$repo_dir"/snapshots/*/ 2>/dev/null | grep -v -- "$2" | head -1 || true)"
  printf '%s' "$out"
}

REPO_DIR="$HF_CACHE/hub/models--${MODEL//\//--}"
SNAP_HOST="$(resolve_snapshot "$MODEL" '-ctprep')"
[ -n "$SNAP_HOST" ] || { echo "!! checkpoint not found under $REPO_DIR - download it first"; exit 1; }
SNAP_NAME="$(basename "$SNAP_HOST")"
DST="$REPO_DIR/snapshots/${SNAP_NAME}-ctprep"
SRC_IN="/hf/hub/models--${MODEL//\//--}/snapshots/${SNAP_NAME}"
DST_IN="${SRC_IN}-ctprep"

DONOR_ARGS=""
if [ "$PLE" = donor ]; then
  DONOR_HOST="$(resolve_snapshot "$PLE_DONOR" '-fp8hybrid')"
  [ -n "$DONOR_HOST" ] || { echo "!! PLE donor $PLE_DONOR not in $HF_CACHE - download it first, or use PLE=keep"; exit 1; }
  DONOR_IN="/hf/hub/models--${PLE_DONOR//\//--}/snapshots/$(basename "$DONOR_HOST")"
  DONOR_ARGS="--donor '$DONOR_IN'"
  echo ">> PLE table will be taken from $PLE_DONOR (verify with tools/verify_ple_donor.py)"
else
  echo ">> PLE table: keeping the checkpoint's own (BF16)"
fi

if [ -f "$DST/.prepared" ]; then echo ">> already prepared: $DST"; exit 0; fi

echo ">> preparing $DST"
docker run --rm --name qwen38-ctprep \
  -v "$HF_CACHE:/hf" -v "$PWD/tools:/tools:ro" --entrypoint bash "$IMAGE" -c "
set -euo pipefail
rm -rf '$DST_IN'
# cp -a keeps the relative symlinks (../../blobs/<sha>) as symlinks: instant, no copy.
cp -a '$SRC_IN' '$DST_IN'
# The two files we rewrite must be real files, not symlinks into the shared blob store.
for f in config.json model.safetensors.index.json; do
  cp --remove-destination \"\$(readlink -f '$DST_IN'/\$f)\" '$DST_IN'/\$f
done
python3 /tools/ct_prepare.py '$DST_IN' --ple '$PLE' $DONOR_ARGS
chmod 644 '$DST_IN'/config.json '$DST_IN'/model.safetensors.index.json
touch '$DST_IN/.prepared'
"
echo ">> ready. Serve with:  MODEL=$MODEL MODE=ct scripts/serve.sh"
