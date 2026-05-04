from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Plot multi-window raw/processed waterfall arrays.")
    p.add_argument("--background-npz", type=Path, required=True)
    p.add_argument("--array-dir", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--percentile", type=float, default=99.5)
    return p.parse_args()


def load_array(path: Path) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(path)
    return np.load(path)


def main() -> None:
    args = parse_args()
    bg = np.load(args.background_npz, allow_pickle=True)
    freqs_mhz = np.asarray(bg["freqs_hz"], dtype=float) / 1e6
    times = np.arange(load_array(args.array_dir / "raw_background.npy").shape[0])
    extent = [float(freqs_mhz.min()), float(freqs_mhz.max()), float(times[0]), float(times[-1])]

    panels = [
        ("raw_background.npy", "Raw background"),
        ("raw_starlink_total.npy", "Injected Starlink total"),
        ("raw_dirty.npy", "Raw dirty = background + Starlink"),
        ("processed_background.npy", "Processed background"),
        ("processed_starlink_only.npy", "Processed Starlink only"),
        ("interaction_residual.npy", "Interaction residual"),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(14.0, 7.2), sharex=True, sharey=True)
    for ax, (fname, title) in zip(axes.ravel(), panels):
        arr = load_array(args.array_dir / fname)
        invalid = ~np.isfinite(arr)
        amp = np.abs(arr)
        amp = np.where(np.isfinite(amp), amp, np.nan)
        vmax = float(np.nanpercentile(amp, args.percentile))
        if not np.isfinite(vmax) or vmax <= 0:
            vmax = float(np.nanmax(amp)) if amp.size else 1.0
        im = ax.imshow(
            np.nan_to_num(amp, nan=0.0, posinf=0.0, neginf=0.0),
            aspect="auto",
            origin="lower",
            extent=extent,
            vmin=0.0,
            vmax=vmax,
            interpolation="nearest",
        )
        if np.any(invalid):
            mask = np.ma.masked_where(~invalid, invalid)
            ax.imshow(
                mask,
                aspect="auto",
                origin="lower",
                extent=extent,
                cmap="Greys",
                vmin=0,
                vmax=1,
                alpha=0.65,
                interpolation="nearest",
            )
        ax.set_title(title)
        ax.set_xlabel("Frequency [MHz]")
        ax.set_ylabel("Time sample")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03, label="|V| [Jy]")

    fig.suptitle(args.array_dir.as_posix())
    fig.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=180, bbox_inches="tight")

    raw = load_array(args.array_dir / "raw_background.npy")
    weights_path = args.array_dir / "weights_tf.npy"
    weights = load_array(weights_path) if weights_path.exists() else None
    invalid = ~np.isfinite(raw)
    rows = np.flatnonzero(np.any(invalid, axis=1))
    cols = np.flatnonzero(np.any(invalid, axis=0))
    report = args.output.with_suffix(".mask_report.txt")
    with open(report, "w", encoding="utf-8") as f:
        f.write(f"raw_background_nan_count={int(np.isnan(raw).sum())}\n")
        f.write(f"raw_background_inf_count={int(np.isinf(raw).sum())}\n")
        if weights is not None:
            f.write(f"weights_zero_count={int((weights == 0).sum())}\n")
            f.write(f"weights_nonfinite_count={int((~np.isfinite(weights)).sum())}\n")
        f.write(f"nan_rows={rows.tolist()}\n")
        f.write(f"nan_cols={cols.tolist()}\n")
        if len(cols):
            f.write(f"nan_freq_mhz_min={float(freqs_mhz[cols].min())}\n")
            f.write(f"nan_freq_mhz_max={float(freqs_mhz[cols].max())}\n")


if __name__ == "__main__":
    main()
