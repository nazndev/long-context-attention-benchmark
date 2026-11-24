"""Load the benchmark YAML and check its shape before any run is expanded."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from spbench.cases import require_valid_config


def load_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open(encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle)
    if not isinstance(loaded, dict):
        raise ValueError(f"{path} did not contain a mapping")
    require_valid_config(loaded)
    return loaded
