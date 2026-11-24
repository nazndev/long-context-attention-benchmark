#!/usr/bin/env bash
# Single-node NCCL launch. This script does not invent a GPU count.
# Example, after the server checks pass:
#   NPROC=2 METHOD=ring SEQ=4096 PHASE=forward_backward bash scripts/run_single_node.sh
set -euo pipefail
if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "nvidia-smi is missing. Refusing to launch a single-node GPU run." >&2
  exit 2
fi
NPROC="${NPROC:-2}"
METHOD="${METHOD:-ring}"
SEQ="${SEQ:-4096}"
PHASE="${PHASE:-forward_backward}"
VARIANT="${VARIANT:-basic}"
EXTRA=()
if [[ "${METHOD}" == "usp" ]]; then
  EXTRA+=(--ulysses-degree "${ULYSSES_DEGREE:?set ULYSSES_DEGREE}" --ring-degree "${RING_DEGREE:?set RING_DEGREE}" --placement "${PLACEMENT:-ulysses_low}")
  if [[ -n "${VARIANT}" ]]; then
    EXTRA+=(--variant "${VARIANT}")
  fi
elif [[ "${METHOD}" == "ring" ]]; then
  EXTRA+=(--variant "${VARIANT}")
fi
exec torchrun --standalone --nproc-per-node="${NPROC}" -m spbench.benchmark \
  --config configs/benchmark.yaml \
  --method "${METHOD}" \
  --seq-len "${SEQ}" \
  --phase "${PHASE}" \
  "${EXTRA[@]}"
