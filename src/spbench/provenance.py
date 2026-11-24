"""Identifiers attached to every experimental record."""

from __future__ import annotations

import hashlib
import importlib.metadata as metadata
import subprocess
from pathlib import Path
from typing import Any

from spbench.upstream import upstream_commit_map

_CACHE: dict[str, Any] | None = None
_HASH_ROOTS = ("src", "tests", "notebooks", "scripts", "configs", "pyproject.toml", "README.md")


def git_commit() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip() or None
    except (OSError, subprocess.CalledProcessError):
        return None


def dependency_versions() -> dict[str, str | None]:
    versions: dict[str, str | None] = {}
    for distribution in ("torch", "PyYAML", "pandas", "matplotlib", "numpy", "pytest"):
        try:
            versions[distribution] = metadata.version(distribution)
        except metadata.PackageNotFoundError:
            versions[distribution] = None
    for distribution in ("ring-flash-attn", "flash-attn", "deepspeed", "yunchang"):
        try:
            versions[distribution] = metadata.version(distribution)
        except metadata.PackageNotFoundError:
            versions[distribution] = None
    versions["lightseq"] = None
    return versions


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def source_tree_sha256() -> str:
    root = _project_root()
    digest = hashlib.sha256()
    files: list[Path] = []
    for name in _HASH_ROOTS:
        path = root / name
        if path.is_file():
            files.append(path)
        elif path.is_dir():
            files.extend(item for item in path.rglob("*") if item.is_file() and "__pycache__" not in item.parts)
    for path in sorted(files):
        relative = path.relative_to(root).as_posix()
        digest.update(relative.encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def config_sha256(config_path: str | Path) -> str:
    return hashlib.sha256(Path(config_path).read_bytes()).hexdigest()


def provenance(config_path: str | Path | None = None) -> dict[str, Any]:
    global _CACHE
    if _CACHE is None:
        _CACHE = {
            "git_commit": git_commit(),
            "source_tree_sha256": source_tree_sha256(),
            "upstream_commits": upstream_commit_map(),
            "dependency_versions": dependency_versions(),
        }
    payload = dict(_CACHE)
    payload["upstream_commits"] = dict(_CACHE["upstream_commits"])
    payload["dependency_versions"] = dict(_CACHE["dependency_versions"])
    if config_path is not None:
        payload["config_sha256"] = config_sha256(config_path)
    return payload


def torch_was_preserved(before: tuple[str | None, str | None], after: tuple[str | None, str | None]) -> bool:
    """True when an existing PyTorch install kept both its version and CUDA build."""

    if before[0] is None:
        return False
    return before == after
