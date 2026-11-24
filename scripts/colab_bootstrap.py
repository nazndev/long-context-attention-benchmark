"""Install this project into the notebook kernel without replacing Colab's PyTorch.

Torch is imported first. If it is already present, pip is not asked to upgrade it.
Optional CUDA packages are not installed here; the notebooks probe them separately
and record the failure. No credentials are read or written.
"""

from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
import zipfile
from pathlib import Path


def in_colab() -> bool:
    return importlib.util.find_spec("google.colab") is not None


def discover_repo(start: str | Path | None = None) -> Path | None:
    """Find the project even when a ZIP extracts one directory level down."""

    root = Path(start or os.getcwd()).resolve()
    pool = [root, *root.parents]
    if root.is_dir():
        for child in root.iterdir():
            if child.is_dir() and child.name not in {".venv", ".git"}:
                pool.append(child)
                for grandchild in child.iterdir():
                    if grandchild.is_dir() and grandchild.name not in {".venv", ".git"}:
                        pool.append(grandchild)
    for candidate in pool:
        if (candidate / "src" / "spbench" / "__init__.py").is_file():
            return candidate
    return None


def materialize_source(repo_url: str = "", commit: str = "", zip_path: str = "", destination: str = "") -> Path:
    """Return a project directory, cloning or unzipping only when a source is given."""

    if zip_path:
        archive = Path(zip_path)
        if not archive.is_file():
            raise FileNotFoundError(f"ZIP_PATH does not exist: {archive}")
        target = Path(destination or "/content/long-context-attention-benchmark")
        target.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(archive) as handle:
            handle.extractall(target)
        found = discover_repo(target)
        if found is None:
            raise FileNotFoundError(f"Unzipped {archive} but src/spbench was not inside {target}")
        return found
    if repo_url:
        if not commit:
            raise ValueError("Set REPO_COMMIT to an immutable commit. A branch name is not enough.")
        target = Path(destination or "/content/long-context-attention-benchmark")
        if not (target / ".git").exists():
            subprocess.check_call(["git", "clone", repo_url, str(target)])
        subprocess.check_call(["git", "checkout", commit], cwd=target)
        found = discover_repo(target)
        if found is None:
            raise FileNotFoundError(f"Checked out {commit} but src/spbench is missing")
        return found
    found = discover_repo()
    if found is None:
        raise FileNotFoundError(
            "No project source found. In Colab, upload the project ZIP and set ZIP_PATH, "
            "or set REPO_URL and REPO_COMMIT. Do not assume a repository is already on the runtime."
        )
    return found


def torch_identity() -> tuple[str | None, str | None]:
    try:
        import torch
    except ImportError:
        return None, None
    return torch.__version__, getattr(torch.version, "cuda", None)


def install_project(project_dir: str | Path) -> None:
    project = Path(project_dir)
    python = sys.executable
    before = torch_identity()
    if before[0] is None:
        raise RuntimeError(
            "PyTorch is not installed in this kernel. On Colab, use a runtime that already has "
            "the CUDA PyTorch build. This installer will not pip-install torch, because the default "
            "wheel can be CPU-only and would replace the GPU runtime."
        )
    subprocess.check_call([python, "-m", "pip", "install", "-e", str(project), "--no-deps"])
    subprocess.check_call(
        [python, "-m", "pip", "install", "PyYAML", "pandas", "matplotlib", "pytest", "--upgrade-strategy", "only-if-needed"]
    )
    after = torch_identity()
    if before != after:
        raise RuntimeError(
            f"PyTorch changed from {before} to {after}. The CUDA installation was not left in place."
        )
    src = str((project / "src").resolve())
    if src not in sys.path:
        sys.path.insert(0, src)
    print(f"Keeping the existing PyTorch {after[0]} (CUDA build: {after[1]}). It was not upgraded.")
