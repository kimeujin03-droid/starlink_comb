from __future__ import annotations

import argparse
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Plot validation figures from nearfield multi-window outputs.")
    p.add_argument("--output-dirs", nargs="+", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    return p.parse_args()


def load_outputs(paths: list[Path]) -> tuple[pd.DataFrame, pd.DataFrame]:
    summaries = []
    nulls = []
    for path in paths:
        tag = path.name
        s_path = path / "tables" / "multi_window_summary.csv"
        n_path = path / "tables" / "phase_randomized_null_ratios.csv"
        if s_path.exists():
            sdf = pd.read_csv(s_path)
            sdf.insert(0, "run", tag)
            summaries.append(sdf)
        if n_path.exists():
            ndf = pd.read_csv(n_path)
            ndf.insert(0, "run", tag)
            nulls.append(ndf)
    return (
        pd.concat(summaries, ignore_index=True) if summaries else pd.DataFrame(),
        pd.concat(nulls, ignore_index=True) if nulls else pd.DataFrame(),
    )


def plot_phase_null_hist(summary: pd.DataFrame, nulls: pd.DataFrame, out: Path) -> None:
    if summary.empty or nulls.empty:
        return
    runs = list(summary["run"].unique())
    n = len(runs)
    fig, axes = plt.subplots(1, n, figsize=(4.2 * n, 3.6), squeeze=False)
    for ax, run in zip(axes.ravel(), runs):
        sdf = summary[summary["run"] == run].iloc[0]
        ndf = nulls[nulls["run"] == run]
        vals = ndf["phase_randomized_coherence_amplification_db"].to_numpy(dtype=float)
        observed = float(sdf["coherence_amplification_db"])
        p95 = float(sdf["phase_randomized_null_ratio_p95"])
        p95_db = 10.0 * math.log10(max(p95, 1e-30))
        ax.hist(vals, bins=32, color="0.72", edgecolor="0.25")
        ax.axvline(observed, color="tab:red", linewidth=2, label="observed")
        ax.axvline(p95_db, color="tab:blue", linestyle="--", linewidth=2, label="null p95")
        ax.set_title(run)
        ax.set_xlabel("Phase-randomized amplification [dB]")
        ax.set_ylabel("Trials")
        ax.grid(alpha=0.25)
    axes.ravel()[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out, dpi=180, bbox_inches="tight")


def plot_local_residual_bars(summary: pd.DataFrame, out: Path) -> None:
    if summary.empty:
        return
    metrics = [
        ("interaction_vs_processed_background_db", "global"),
        ("local_eor_window_residual_vs_processed_background_db", "EoR window"),
        ("local_peak_delay_bin_excess_db", "local peak"),
        ("top_1_percent_delay_bin_excess_db", "top 1%"),
        ("top_5_percent_delay_bin_excess_db", "top 5%"),
    ]
    fig, ax = plt.subplots(figsize=(max(7.0, 1.2 * len(summary) * len(metrics)), 4.2))
    x = np.arange(len(summary))
    width = 0.16
    for k, (col, label) in enumerate(metrics):
        if col in summary.columns:
            ax.bar(x + (k - 2) * width, pd.to_numeric(summary[col], errors="coerce"), width=width, label=label)
    ax.axhline(0, color="k", linewidth=1)
    ax.set_xticks(x)
    ax.set_xticklabels(summary["run"], rotation=30, ha="right")
    ax.set_ylabel("Residual / processed background [dB]")
    ax.set_title("Global vs local delay-bin residual budget")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out, dpi=180, bbox_inches="tight")


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary, nulls = load_outputs(args.output_dirs)
    if not summary.empty:
        summary.to_csv(args.output_dir / "paper_validation_summary.csv", index=False)
    if not nulls.empty:
        nulls.to_csv(args.output_dir / "paper_validation_phase_nulls.csv", index=False)
    plot_phase_null_hist(summary, nulls, args.output_dir / "fig_phase_randomized_null_histograms.png")
    plot_local_residual_bars(summary, args.output_dir / "fig_local_residual_budget_bars.png")


if __name__ == "__main__":
    main()
