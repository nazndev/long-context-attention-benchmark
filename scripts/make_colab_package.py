"""Build spbench_colab.zip for a fresh Colab upload.

The archive contains the source, notebooks, tests, and docs required to run
notebooks 01 and 02. It omits the virtualenv, the macOS CPU lockfile, internal
instruction files, and local measurement logs.
"""

from __future__ import annotations

import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "spbench_colab.zip"
INCLUDE_ROOTS = (
    "src",
    "tests",
    "notebooks",
    "scripts",
    "configs",
    "docs",
    "reports",
    "README.md",
    "STATUS.md",
    "pyproject.toml",
    ".gitignore",
)
EXCLUDED_PARTS = {
    ".venv",
    ".git",
    "__pycache__",
    ".pytest_cache",
    ".ipynb_checkpoints",
    "src/spbench.egg-info",
}
EXCLUDED_NAMES = {
    "requirements-cpu-dev.txt",
    "CSE598_Project_05_Colab_Research_Plan_v3.yml",
}


def should_pack(relative: str) -> bool:
    path = Path(relative)
    if path.name in EXCLUDED_NAMES or path.name.startswith("CSE598_Project_") and "Research_Plan" in path.name:
        return False
    if "Colab_Research_Plan" in path.name or "cursor_instructions" in path.name:
        return False
    parts = set(path.parts)
    if parts & EXCLUDED_PARTS:
        return False
    if path.suffix in {".pyc", ".zip"}:
        return False
    if relative.startswith("results/raw/") or relative.startswith("results/processed/"):
        return False
    return True


def build(destination: Path = OUTPUT) -> Path:
    with zipfile.ZipFile(destination, "w", zipfile.ZIP_DEFLATED) as archive:
        for name in INCLUDE_ROOTS:
            path = ROOT / name
            files = [path] if path.is_file() else [item for item in path.rglob("*") if item.is_file()]
            for item in files:
                relative = item.relative_to(ROOT).as_posix()
                if should_pack(relative):
                    archive.write(item, relative)
    return destination


if __name__ == "__main__":
    archive = build()
    print(archive, archive.stat().st_size)
