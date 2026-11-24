"""Analytical demonstrations used by the notebooks."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from spbench.plots import plot_analytical_causal_imbalance
from spbench.work_model import explain_not_constant_flops, imbalance_ratios, query_owned_pairs


def causal_imbalance_frame(sequence_length: int = 1024, world_size: int = 8) -> pd.DataFrame:
    rows = []
    for layout, variant in (("contiguous", "contiguous"), ("zigzag", "zigzag")):
        for rank in range(world_size):
            rows.append(
                {
                    "origin": "analytical",
                    "layout": layout,
                    "rank": rank,
                    "valid_causal_pairs": query_owned_pairs(sequence_length, world_size, rank, variant),
                    "sequence_length": sequence_length,
                    "world_size": world_size,
                }
            )
    return pd.DataFrame(rows)


def write_causal_imbalance_figure(path: str | Path, sequence_length: int = 1024, world_size: int = 8) -> dict:
    frame = causal_imbalance_frame(sequence_length, world_size)
    summary = {}
    for layout, group in frame.groupby("layout"):
        summary[layout] = imbalance_ratios(group["valid_causal_pairs"].astype(float).tolist())
    written = plot_analytical_causal_imbalance(frame, path)
    return {
        "path": str(path),
        "sequence_length": sequence_length,
        "world_size": world_size,
        "imbalance": summary,
        "plotted": written["plotted"],
        "weak_context_note": explain_not_constant_flops(4096, world_size),
    }
