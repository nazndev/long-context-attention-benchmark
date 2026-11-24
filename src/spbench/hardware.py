"""Runtime detection."""

from __future__ import annotations

import platform
import shutil
import subprocess
import sys
from typing import Any


_ARCHITECTURES = {
    (7, 0): "Volta",
    (7, 5): "Turing",
    (8, 0): "Ampere",
    (8, 6): "Ampere",
    (8, 9): "Ada",
    (9, 0): "Hopper",
    (10, 0): "Blackwell",
}


def architecture_name(major: int, minor: int) -> str:
    known = _ARCHITECTURES.get((major, minor))
    if known is None:
        return f"unlisted sm_{major}{minor}"
    return known


def _run(command: list[str]) -> str | None:
    if shutil.which(command[0]) is None:
        return None
    try:
        completed = subprocess.run(command, check=False, capture_output=True, text=True, timeout=20)
    except (OSError, subprocess.TimeoutExpired):
        return None
    text = (completed.stdout or "") + (completed.stderr or "")
    return text.strip() or None


def detect_hardware() -> dict[str, Any]:
    info: dict[str, Any] = {
        "python": sys.version.split()[0],
        "executable": sys.executable,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "system": platform.system(),
        "torch_version": None,
        "torch_cuda_build": None,
        "cuda_available": False,
        "cuda_device_count": 0,
        "gpu_names": [],
        "compute_capabilities": [],
        "gpu_architectures": [],
        "gpu_memory_mib": [],
        "gpu_free_mib": [],
        "bf16_hardware": False,
        "mps_available": False,
        "nccl_available": False,
        "nvidia_smi": None,
        "nvidia_smi_topo": None,
        "two_node_100gbe_verified": False,
        "hypothetical": False,
        "notes": [],
    }
    try:
        import torch
    except ImportError:
        info["notes"].append("PyTorch is not importable in this interpreter.")
        torch = None
    if torch is not None:
        info["torch_version"] = torch.__version__
        info["torch_cuda_build"] = torch.version.cuda
        info["cuda_available"] = bool(torch.cuda.is_available())
        if info["cuda_available"]:
            info["cuda_device_count"] = int(torch.cuda.device_count())
            for index in range(info["cuda_device_count"]):
                props = torch.cuda.get_device_properties(index)
                info["gpu_names"].append(props.name)
                info["compute_capabilities"].append(f"{props.major}.{props.minor}")
                info["gpu_architectures"].append(architecture_name(props.major, props.minor))
                info["gpu_memory_mib"].append(int(props.total_memory // (1024 * 1024)))
                free_bytes, _total_bytes = torch.cuda.mem_get_info(index)
                info["gpu_free_mib"].append(int(free_bytes // (1024 * 1024)))
            bf16 = getattr(torch.cuda, "is_bf16_supported", None)
            info["bf16_hardware"] = bool(bf16()) if callable(bf16) else False
        mps = getattr(getattr(torch, "backends", None), "mps", None)
        info["mps_available"] = bool(mps is not None and mps.is_available())
        nccl_probe = getattr(torch.distributed, "is_nccl_available", None)
        info["nccl_available"] = bool(nccl_probe()) if callable(nccl_probe) else False
    info["nvidia_smi"] = _run(["nvidia-smi", "-L"])
    info["nvidia_smi_topo"] = _run(["nvidia-smi", "topo", "-m"])
    if info["mps_available"] and not info["cuda_available"]:
        info["notes"].append(
            "An Apple GPU is visible through MPS. It is not an NVIDIA CUDA device and is not used for timed baselines."
        )
    if not info["cuda_available"]:
        info["notes"].append("No CUDA device is available to this process.")
    if not info["two_node_100gbe_verified"]:
        info["notes"].append("Two-server 100 Gb Ethernet has not been verified in this environment.")
    return info


def hypothetical_server_hardware() -> dict[str, Any]:
    """The allocation requested in the preliminary report. This is not a detected machine."""

    return {
        "python": None,
        "executable": None,
        "platform": "hypothetical-linux",
        "machine": "x86_64",
        "processor": None,
        "system": "Linux",
        "torch_version": None,
        "torch_cuda_build": None,
        "cuda_available": True,
        "cuda_device_count": 8,
        "gpu_names": ["hypothetical-matched-nvidia-gpu"] * 8,
        "compute_capabilities": [],
        "gpu_architectures": [],
        "gpu_memory_mib": [],
        "gpu_free_mib": [],
        "bf16_hardware": True,
        "mps_available": False,
        "nccl_available": True,
        "nvidia_smi": None,
        "nvidia_smi_topo": None,
        "two_node_100gbe_verified": True,
        "hypothetical": True,
        "notes": [
            "Planning profile only. No process observed these GPUs, this NCCL build, or a 100 Gb Ethernet link."
        ],
    }
