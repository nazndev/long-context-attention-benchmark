"""Import probes. A failed import is an unresolved dependency, not a fallback."""

from __future__ import annotations

import importlib


MODULES = {
    "torch": "torch",
    "ring_flash_attn": "ring_flash_attn",
    "lightseq": "lightseq_async_attn",
    "deepspeed": "deepspeed",
    "yunchang": "yunchang",
    "flash_attn": "flash_attn",
}


def probe_dependencies() -> dict[str, bool]:
    found = {}
    for name, module in MODULES.items():
        try:
            importlib.import_module(module)
        except Exception:
            found[name] = False
        else:
            found[name] = True
    return found
