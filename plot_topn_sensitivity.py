from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Plot top-N emitter sensitivity.")
    p.add_argument("--summaries", nargs="+", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    rows = []
    for path in args.summaries:
        df = pd.read_csv(path)
        window = path.parent.name.replace("topn_sensitivity_", "")
        df.insert(0, "window", window)
        rows.append(df)
    out = pd.concat(rows, ignore_index=True)
    out["top_N"] = out["case"].str.replace("top", "", regex=False).astype(int)
    out["null_p95_db"] = 10.0 * (out["phase_randomized_null_ratio_p95"].astype(float)).map(lambda x: __import__("math").log10(max(x, 1e-30)))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.output_dir / "topn_sensitivity_summary.csv", index=False)

    fig, axes = plt.subplots(1, 3, figsize=(12.0, 3.8), sharex=True)
    metrics = [
        ("coherence_amplification_db", "Observed amplification [dB]"),
        ("null_p95_db", "Phase-null p95 [dB]"),
        ("observed_minus_phase_null_median_db", "Observed - null median [dB]"),
    ]
    for ax, (col, label) in zip(axes, metrics):
        for window, sub in out.sort_values("top_N").groupby("window"):
            ax.plot(sub["top_N"], pd.to_numeric(sub[col], errors="coerce"), marker="o", label=window)
        ax.set_xlabel("Top-N emitters")
        ax.set_ylabel(label)
        ax.grid(alpha=0.3)
    axes[0].legend(fontsize=8)
    fig.suptitle("Top-N sensitivity with fixed predicted-apparent-flux selection")
    fig.tight_layout()
    fig.savefig(args.output_dir / "fig_topn_sensitivity.png", dpi=180, bbox_inches="tight")


if __name__ == "__main__":
    main()
