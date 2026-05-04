from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Plot unpolarized vs anti-correlated EE/NN contrast sweep.")
    p.add_argument("--summary", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    df = pd.read_csv(args.summary)
    df["contrast"] = pd.to_numeric(df["polarization_contrast"], errors="coerce").fillna(0.0)
    df["ratio"] = pd.to_numeric(df["coherence_amplification_ratio"], errors="coerce")
    df["local_peak_db"] = pd.to_numeric(df["local_peak_delay_bin_excess_db"], errors="coerce")
    df["eor_db"] = pd.to_numeric(df["local_eor_window_residual_vs_processed_background_db"], errors="coerce")
    df["q_over_i"] = pd.to_numeric(df["polarization_stokes_Q_over_I_mean_abs"], errors="coerce")

    fig, axes = plt.subplots(2, 2, figsize=(10.5, 7.0), sharex=True)
    metrics = [
        ("ratio", "Coherence amplification ratio"),
        ("local_peak_db", "Local peak delay-bin excess [dB]"),
        ("eor_db", "EoR-window residual / background [dB]"),
        ("q_over_i", "Mean |Q|/|I|"),
    ]
    for ax, (col, label) in zip(axes.ravel(), metrics):
        for pol, sub in df.sort_values("contrast").groupby("selected_pol"):
            anti = sub[sub["polarization_mode"] == "jones_anti_correlated"]
            ax.plot(anti["contrast"], anti[col], marker="o", label=f"{pol} anti")
            unpol = sub[sub["polarization_mode"] == "jones_unpolarized"]
            if not unpol.empty:
                ax.scatter([0.0], [float(unpol[col].iloc[0])], marker="x", s=60, label=f"{pol} unpol")
        ax.set_ylabel(label)
        ax.grid(alpha=0.3)
    for ax in axes[-1, :]:
        ax.set_xlabel("Anti-correlated contrast")
    axes[0, 0].legend(fontsize=8)
    fig.suptitle("Jones polarization contrast sweep: unpolarized vs anti-correlated EE/NN")
    fig.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=180, bbox_inches="tight")


if __name__ == "__main__":
    main()
