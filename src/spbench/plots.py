"""Figure writers."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from spbench.errors import PlotDataError


def _is_measured_path(path: Path) -> bool:
    parts = path.resolve().parts
    return "measured" in parts and "figures" in parts


def _require_measured(frame: pd.DataFrame, path: Path) -> pd.DataFrame:
    if _is_measured_path(path):
        if frame.empty or "origin" not in frame.columns or not (frame["origin"] == "measured").all():
            raise PlotDataError("measured figures require origin='measured' on every row")
        if not (frame["status"] == "success").all():
            raise PlotDataError("measured figures cannot contain unsuccessful rows; withhold them instead of plotting zeros")
    return frame


def _save(figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, bbox_inches="tight")
    plt.close(figure)


def plot_analytical_causal_imbalance(frame: pd.DataFrame, path: str | Path) -> dict[str, int]:
    """Bar chart of query-owned causal pairs. The frame is analytical, not timed."""

    destination = Path(path)
    if _is_measured_path(destination):
        raise PlotDataError("analytical imbalance cannot be written into the measured figure directory")
    if frame.empty or not (frame["origin"] == "analytical").all():
        raise PlotDataError("analytical imbalance plot requires origin='analytical'")
    figure, axis = plt.subplots(figsize=(8, 4.5))
    for label, group in frame.groupby("layout"):
        axis.plot(group["rank"], group["valid_causal_pairs"], marker="o", label=label)
    axis.set_xlabel("Rank")
    axis.set_ylabel("Valid causal pairs per head and batch")
    axis.set_title("Analytical causal imbalance (not a GPU measurement)")
    axis.legend()
    _save(figure, destination)
    return {"plotted": int(len(frame)), "withheld": 0}


def _split(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    success = frame[frame["status"] == "success"].copy()
    withheld = frame[frame["status"] != "success"].copy()
    return success, withheld


def _phase_title(base: str, frame: pd.DataFrame) -> str:
    if "phase" in frame.columns and frame["phase"].nunique() == 1:
        return f"{base} ({frame['phase'].iloc[0]})"
    return base


def plot_throughput(frame: pd.DataFrame, path: str | Path) -> dict[str, int]:
    destination = Path(path)
    success, withheld = _split(frame)
    _require_measured(success, destination)
    figure, axis = plt.subplots(figsize=(8, 4.5))
    if not success.empty:
        columns = ["method", "topology"] + (["phase"] if "phase" in success.columns else [])
        for key, group in success.groupby(columns):
            ordered = group.sort_values("global_sequence_length")
            label = " / ".join(str(part) for part in (key if isinstance(key, tuple) else (key,)))
            axis.plot(
                ordered["global_sequence_length"],
                ordered["global_tokens_per_second"],
                marker="o",
                label=label,
            )
    axis.set_xlabel("Global sequence length (tokens)")
    axis.set_ylabel("Throughput (global tokens / s)")
    axis.set_title(_phase_title("Measured throughput", success))
    if not success.empty:
        axis.legend()
    _save(figure, destination)
    return {"plotted": int(len(success)), "withheld": int(len(withheld))}


def plot_memory(frame: pd.DataFrame, path: str | Path) -> dict[str, int]:
    destination = Path(path)
    success, withheld = _split(frame)
    _require_measured(success, destination)
    figure, axis = plt.subplots(figsize=(8, 4.5))
    if not success.empty:
        columns = ["method"] + (["phase"] if "phase" in success.columns else [])
        for key, group in success.groupby(columns):
            ordered = group.sort_values("global_sequence_length")
            label = " / ".join(str(part) for part in (key if isinstance(key, tuple) else (key,)))
            axis.plot(ordered["global_sequence_length"], ordered["peak_allocated_mib"], marker="o", label=f"{label} allocated")
            if "peak_reserved_mib" in ordered:
                axis.plot(
                    ordered["global_sequence_length"],
                    ordered["peak_reserved_mib"],
                    marker="x",
                    linestyle="--",
                    label=f"{label} reserved",
                )
    axis.set_xlabel("Global sequence length (tokens)")
    axis.set_ylabel("Maximum per-GPU memory (MiB)")
    axis.set_title(_phase_title("Measured peak GPU memory", success))
    if not success.empty:
        axis.legend()
    _save(figure, destination)
    return {"plotted": int(len(success)), "withheld": int(len(withheld))}


def plot_strong_scaling(frame: pd.DataFrame, path: str | Path) -> dict[str, int]:
    destination = Path(path)
    success, withheld = _split(frame)
    _require_measured(success, destination)
    figure, axis = plt.subplots(figsize=(8, 4.5))
    if not success.empty and "scaling_efficiency" in success.columns:
        for method, group in success.groupby("method"):
            ordered = group.sort_values("world_size")
            axis.plot(ordered["world_size"], ordered["scaling_efficiency"], marker="o", label=method)
    axis.set_xlabel("GPU count")
    axis.set_ylabel("Strong-scaling efficiency")
    axis.set_title("Measured strong scaling at fixed global length")
    if not success.empty:
        axis.legend()
    _save(figure, destination)
    return {"plotted": int(len(success)), "withheld": int(len(withheld))}


def plot_category_bars(frame: pd.DataFrame, path: str | Path, value: str, ylabel: str, title: str) -> dict[str, int]:
    destination = Path(path)
    success, withheld = _split(frame)
    _require_measured(success, destination)
    figure, axis = plt.subplots(figsize=(8, 4.5))
    if not success.empty:
        labels = success["label"].astype(str).tolist()
        axis.bar(labels, success[value])
        axis.tick_params(axis="x", rotation=30)
    axis.set_ylabel(ylabel)
    axis.set_title(title)
    _save(figure, destination)
    return {"plotted": int(len(success)), "withheld": int(len(withheld))}
