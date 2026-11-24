"""Gloo process-group smoke test."""

from __future__ import annotations

import os

import torch.distributed as dist


def main() -> int:
    backend = os.environ.get("SPBENCH_SMOKE_BACKEND", "gloo")
    if backend != "gloo":
        print("This smoke test only runs Gloo. NCCL is a GPU-server check and is not implied here.")
        return 2
    dist.init_process_group(backend="gloo")
    try:
        rank = dist.get_rank()
        world = dist.get_world_size()
        dist.barrier()
        print(f"gloo_smoke rank={rank} world={world}")
        return 0
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    raise SystemExit(main())
