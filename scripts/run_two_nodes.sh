#!/usr/bin/env bash
# Two-node launch. Fill the approved address and port. Do not point this at a Colab session.
# Example on each server:
#   NNODES=2 NPROC=2 NODE_RANK=0 MASTER_ADDR=10.0.0.1 MASTER_PORT=29500 \
#   METHOD=usp ULYSSES_DEGREE=2 RING_DEGREE=2 bash scripts/run_two_nodes.sh
set -euo pipefail
if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "nvidia-smi is missing. Refusing to launch a two-node GPU run." >&2
  exit 2
fi
: "${MASTER_ADDR:?set MASTER_ADDR to the approved node 0 address}"
: "${MASTER_PORT:?set MASTER_PORT to the approved port}"
: "${NODE_RANK:?set NODE_RANK to 0 or 1}"
NPROC="${NPROC:-2}"
METHOD="${METHOD:-usp}"
SEQ="${SEQ:-4096}"
PHASE="${PHASE:-forward_backward}"
export SPBENCH_NODES=2
if [[ -z "${NCCL_SOCKET_IFNAME:-}" ]]; then
  echo "NCCL_SOCKET_IFNAME is unset. Set it to the verified 100 GbE interface before comparing topologies." >&2
  exit 2
fi
EXTRA=()
if [[ "${METHOD}" == "usp" ]]; then
  EXTRA+=(--variant "${VARIANT:-basic}" --ulysses-degree "${ULYSSES_DEGREE:?set ULYSSES_DEGREE}" --ring-degree "${RING_DEGREE:?set RING_DEGREE}" --placement "${PLACEMENT:-ulysses_low}")
elif [[ "${METHOD}" == "ring" ]]; then
  EXTRA+=(--variant "${VARIANT:-basic}")
fi
exec torchrun --nnodes=2 --nproc-per-node="${NPROC}" --node-rank="${NODE_RANK}" \
  --master-addr="${MASTER_ADDR}" --master-port="${MASTER_PORT}" \
  -m spbench.benchmark \
  --config configs/benchmark.yaml \
  --method "${METHOD}" \
  --seq-len "${SEQ}" \
  --phase "${PHASE}" \
  "${EXTRA[@]}"
