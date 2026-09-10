#!/usr/bin/env bash
# One-time graft of the NVFP4 MTP draft experts onto a prepared compressed-tensors
# checkpoint -- the MODE=ct counterpart of scripts/prepare-mtp-graft.sh.
#
#   MODEL=<org/name> scripts/prepare-ct-mtp.sh     # needs scripts/prepare-ct.sh first
#   MODEL=<org/name> MODE=ct-mtp scripts/serve.sh
#
# The draft head ships its routed experts as fused BF16 (~5 GB). Inferact publishes
# the same experts in NVFP4 (~1.4 GB), so the swap frees ~3.4 GiB. Note where that
# memory goes: GPU_MEM sizes vLLM's arena as a fraction of TOTAL device memory,
# independent of how large the weights are, so freeing weight memory grows the KV
# cache and returns nothing to the system. If you would rather have it as page
# cache and headroom, pin the KV pool with KV_CACHE_MEM and let the rest fall back.
#
# Two things differ from the ModelOpt graft, both handled in tools/ct_mtp_graft.py:
# the donor is ModelOpt-format and has to be transcoded to compressed-tensors names
# (with weight_scale_2 inverted into weight_global_scale), and the config surgery is
# a single ignore-entry removal rather than a 29-name replacement, because
# compressed-tensors targets its routed experts by a regex that matches the MTP
# experts and nothing else under mtp. See that file for the reasoning and evidence.
#
# Layout: a sibling of the -ctprep snapshot, made of the same relative symlinks --
# only the rewritten MTP shard, the new NVFP4 expert shard, config.json and the
# index are real files. Neither parent snapshot is touched.
set -euo pipefail

MODEL="${MODEL:?set MODEL to the compressed-tensors repo, e.g. org/Name}"
IMAGE="${IMAGE:-qwen38-flash-dgx}"
HF_CACHE="${HF_CACHE:-$HOME/.cache/huggingface}"

# Pinned exactly as scripts/prepare-mtp-graft.sh pins it: a different donor
# revision needs re-checking, not just re-downloading.
DONOR_REPO="Inferact/Qwen3.8-Flash-Next-NVFP4"
DONOR_REV="103a7608316173ca6edd49929544244de7ffda70"
DONOR_SHA="0d44e6d705d2313c713e60114e56874adf358ed5f646dc8704bb5be15f5ddbf7"
DONOR_SHARD="nvfp4_experts_mtp.safetensors"

REPO_DIR="$HF_CACHE/hub/models--${MODEL//\//--}"
SNAP_HOST=""
for REF in main master; do
  REV="$(cat "$REPO_DIR/refs/$REF" 2>/dev/null || true)"
  if [ -n "$REV" ] && [ -d "$REPO_DIR/snapshots/$REV" ]; then SNAP_HOST="$REPO_DIR/snapshots/$REV/"; break; fi
done
SNAP_HOST="${SNAP_HOST:-$(ls -dt "$REPO_DIR"/snapshots/*/ 2>/dev/null | grep -v -e '-ctprep' | head -1 || true)}"
[ -n "$SNAP_HOST" ] || { echo "!! checkpoint not found under $REPO_DIR"; exit 1; }
SNAP_NAME="$(basename "$SNAP_HOST")"
CTPREP="$REPO_DIR/snapshots/${SNAP_NAME}-ctprep"
[ -f "$CTPREP/.prepared" ] || {
  echo "!! run scripts/prepare-ct.sh first (one-time)"; exit 1; }
GRAFT="${CTPREP}-mtpnvfp4"
[ -f "$GRAFT/.prepared" ] && { echo ">> already prepared: $GRAFT"; exit 0; }

DONOR_DIR="$HF_CACHE/hub/models--${DONOR_REPO//\//--}"
CTPREP_IN="/hf/hub/models--${MODEL//\//--}/snapshots/${SNAP_NAME}-ctprep"
DONOR_BLOB_IN="/hf/hub/models--${DONOR_REPO//\//--}/blobs/$DONOR_SHA"

echo ">> grafting NVFP4 MTP experts onto $(basename "$CTPREP")"
docker run --rm --name qwen38-ctmtp \
  -v "$HF_CACHE:/hf" -v "$PWD/tools:/tools:ro" --entrypoint bash "$IMAGE" -c "
set -euo pipefail
BLOB='$DONOR_BLOB_IN'
if [ ! -f \"\$BLOB\" ]; then
  echo '>> downloading donor shard (~1.5 GiB, pinned revision + sha256)'
  mkdir -p \"\$(dirname \"\$BLOB\")\" '/hf/hub/models--${DONOR_REPO//\//--}/refs'
  echo '$DONOR_REV' > '/hf/hub/models--${DONOR_REPO//\//--}/refs/main'
  curl -fL --retry 3 -o \"\$BLOB.incomplete\" \\
    'https://huggingface.co/$DONOR_REPO/resolve/$DONOR_REV/$DONOR_SHARD'
  got=\$(sha256sum \"\$BLOB.incomplete\" | cut -d' ' -f1)
  [ \"\$got\" = '$DONOR_SHA' ] || { rm -f \"\$BLOB.incomplete\"; echo \"!! donor sha256 mismatch: \$got\"; exit 1; }
  mv \"\$BLOB.incomplete\" \"\$BLOB\"
else
  echo '>> donor shard already in the cache'
fi
python3 /tools/ct_mtp_graft.py '$CTPREP_IN' --donor \"\$BLOB\"
chmod -R a+rX '${CTPREP_IN}-mtpnvfp4'
"
echo ">> ready. Serve with:  MODEL=$MODEL MODE=ct-mtp scripts/serve.sh"
