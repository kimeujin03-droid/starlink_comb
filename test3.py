import json
import math
import os
from dataclasses import dataclass, asdict
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd


# ============================================================
# Config
# ============================================================

RESULT_JSON = "comb_results.json"
ANNOTATION_CSV = "experiment2_annotations.csv"
OUTDIR = "exp3_boundary_calibration"

RANDOM_SEED = 42
CALIBRATION_FRACTION = 0.7
N_BOOT = 500

# spacing mode clustering tolerance (MHz)
SPACING_CLUSTER_EPS = 0.20

# near-boundary margin as fraction of bin width
BOUNDARY_MARGIN_FRAC = 0.15


# ============================================================
# Utilities
# ============================================================

def ensure_outdir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def safe_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def qcut_boundaries(values: np.ndarray, probs: List[float]) -> List[float]:
    """Return quantile boundaries, ignoring NaN."""
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return []
    return [float(np.quantile(values, p)) for p in probs]


def nearest_boundary_distance(x: float, boundaries: List[float]) -> float:
    if not np.isfinite(x) or len(boundaries) == 0:
        return np.nan
    return min(abs(x - b) for b in boundaries)


def empirical_mode_centers(values: np.ndarray, eps: float = 0.2) -> List[float]:
    """
    Very simple 1D mode finder by greedy clustering.
    Sort values, group nearby points within eps, return cluster means.
    """
    values = np.sort(values[np.isfinite(values)])
    if len(values) == 0:
        return []

    clusters = []
    current = [values[0]]

    for v in values[1:]:
        if abs(v - np.mean(current)) <= eps:
            current.append(v)
        else:
            clusters.append(current)
            current = [v]
    clusters.append(current)

    centers = [float(np.mean(c)) for c in clusters]
    counts = [len(c) for c in clusters]

    # sort by count desc, then by center asc
    ranked = sorted(zip(centers, counts), key=lambda x: (-x[1], x[0]))
    return [c for c, _ in ranked]


def assign_spacing_class(x: float, boundaries: List[float]) -> Optional[int]:
    """
    boundaries define bins:
      class 0: x < b1
      class 1: b1 <= x < b2
      ...
      class k: x >= last boundary
    """
    if not np.isfinite(x):
        return None
    for i, b in enumerate(boundaries):
        if x < b:
            return i
    return len(boundaries)


def assign_quantile_class(x: float, boundaries: List[float]) -> Optional[int]:
    return assign_spacing_class(x, boundaries)


def class_code(prefix: str, idx: Optional[int]) -> str:
    if idx is None:
        return f"{prefix}NA"
    return f"{prefix}{idx+1}"


@dataclass
class BoundarySet:
    spacing_mode_centers: List[float]
    spacing_boundaries: List[float]
    jitter_boundaries: List[float]
    occupancy_boundaries: List[float]
    dwell_boundaries: List[float]

    def to_dict(self) -> Dict:
        return asdict(self)


# ============================================================
# Data loading
# ============================================================

def load_results(result_json: str) -> pd.DataFrame:
    with open(result_json, "r", encoding="utf-8") as f:
        results = json.load(f)
    df = pd.DataFrame(results)

    # normalize expected columns
    for col in ["delta_f_hat_mhz", "jitter_mhz", "O", "dwell_med"]:
        if col in df.columns:
            df[col] = safe_numeric(df[col])

    # build sample_id if absent
    if "sample_id" not in df.columns:
        needed = {"file", "pol", "band_class"}
        if needed.issubset(df.columns):
            df["sample_id"] = (
                df["file"].astype(str)
                + "|"
                + df["pol"].astype(str)
                + "|"
                + df["band_class"].astype(str)
            )
        else:
            raise ValueError("Need sample_id or (file, pol, band_class) columns.")
    return df


def load_annotations(annotation_csv: str) -> pd.DataFrame:
    ann = pd.read_csv(annotation_csv)

    if "sample_id" not in ann.columns:
        needed = {"file", "pol", "band_class"}
        if needed.issubset(ann.columns):
            ann["sample_id"] = (
                ann["file"].astype(str)
                + "|"
                + ann["pol"].astype(str)
                + "|"
                + ann["band_class"].astype(str)
            )
        else:
            raise ValueError("Annotation CSV needs sample_id or (file, pol, band_class).")
    return ann


def merge_results_annotations(df: pd.DataFrame, ann: pd.DataFrame) -> pd.DataFrame:
    merged = df.merge(
        ann[
            [
                "sample_id",
                "benchmark_split",
                "annotation_label",
                "artifact_type",
                "notes",
            ]
        ],
        on="sample_id",
        how="inner",
    )
    return merged


# ============================================================
# Step 1. Calibration boundaries
# ============================================================

def propose_boundaries(calib_df: pd.DataFrame) -> BoundarySet:
    # Use only finite spacing detections for spacing mode proposal
    spacing = calib_df["delta_f_hat_mhz"].to_numpy()
    mode_centers = empirical_mode_centers(spacing, eps=SPACING_CLUSTER_EPS)

    # spacing boundaries are midpoints between dominant mode centers
    mode_centers_sorted = sorted(mode_centers)
    spacing_boundaries = []
    for a, b in zip(mode_centers_sorted[:-1], mode_centers_sorted[1:]):
        spacing_boundaries.append(float((a + b) / 2.0))

    # For J / O / dwell, use tercile-like proposal from calibration positives
    jitter = calib_df["jitter_mhz"].to_numpy()
    occupancy = calib_df["O"].to_numpy()
    dwell = calib_df["dwell_med"].to_numpy()

    jitter_boundaries = qcut_boundaries(jitter, [1/3, 2/3])
    occupancy_boundaries = qcut_boundaries(occupancy, [1/3, 2/3])

    # dwell may be mostly NaN in your current pilot
    if np.isfinite(dwell).sum() >= 5:
        dwell_boundaries = qcut_boundaries(dwell, [1/3, 2/3])
    else:
        dwell_boundaries = []

    return BoundarySet(
        spacing_mode_centers=mode_centers_sorted,
        spacing_boundaries=spacing_boundaries,
        jitter_boundaries=jitter_boundaries,
        occupancy_boundaries=occupancy_boundaries,
        dwell_boundaries=dwell_boundaries,
    )


# ============================================================
# Label assignment
# ============================================================

def apply_frozen_boundaries(df: pd.DataFrame, bounds: BoundarySet) -> pd.DataFrame:
    out = df.copy()

    out["spacing_class_idx"] = out["delta_f_hat_mhz"].apply(
        lambda x: assign_spacing_class(x, bounds.spacing_boundaries)
    )
    out["jitter_class_idx"] = out["jitter_mhz"].apply(
        lambda x: assign_quantile_class(x, bounds.jitter_boundaries)
    )
    out["occupancy_class_idx"] = out["O"].apply(
        lambda x: assign_quantile_class(x, bounds.occupancy_boundaries)
    )

    # dwell may be absent or too sparse
    if len(bounds.dwell_boundaries) > 0 and "dwell_med" in out.columns:
        out["dwell_class_idx"] = out["dwell_med"].apply(
            lambda x: assign_quantile_class(x, bounds.dwell_boundaries)
        )
    else:
        out["dwell_class_idx"] = None

    out["spacing_code"] = out["spacing_class_idx"].apply(lambda x: class_code("DF", x))
    out["jitter_code"] = out["jitter_class_idx"].apply(lambda x: class_code("J", x))
    out["occupancy_code"] = out["occupancy_class_idx"].apply(lambda x: class_code("O", x))

    out["frozen_label"] = (
        out["spacing_code"].astype(str)
        + "-"
        + out["jitter_code"].astype(str)
        + "-"
        + out["occupancy_code"].astype(str)
    )

    return out


def compute_near_boundary_flags(df: pd.DataFrame, bounds: BoundarySet) -> pd.DataFrame:
    out = df.copy()

    spacing_widths = []
    full_spacing_edges = [-np.inf] + bounds.spacing_boundaries + [np.inf]
    for a, b in zip(full_spacing_edges[:-1], full_spacing_edges[1:]):
        if np.isfinite(a) and np.isfinite(b):
            spacing_widths.append(b - a)

    mean_spacing_width = np.nanmean(spacing_widths) if spacing_widths else np.nan
    spacing_margin = (
        mean_spacing_width * BOUNDARY_MARGIN_FRAC if np.isfinite(mean_spacing_width) else np.nan
    )

    out["dist_to_spacing_boundary"] = out["delta_f_hat_mhz"].apply(
        lambda x: nearest_boundary_distance(x, bounds.spacing_boundaries)
    )
    out["near_spacing_boundary"] = (
        out["dist_to_spacing_boundary"] <= spacing_margin
        if np.isfinite(spacing_margin)
        else False
    )

    out["dist_to_jitter_boundary"] = out["jitter_mhz"].apply(
        lambda x: nearest_boundary_distance(x, bounds.jitter_boundaries)
    )
    if len(bounds.jitter_boundaries) > 0:
        jitter_values = out["jitter_mhz"].dropna().to_numpy()
        jitter_width = np.nan
        if len(jitter_values) > 1:
            jitter_width = (np.nanmax(jitter_values) - np.nanmin(jitter_values)) / 3.0
        jitter_margin = jitter_width * BOUNDARY_MARGIN_FRAC if np.isfinite(jitter_width) else np.nan
        out["near_jitter_boundary"] = (
            out["dist_to_jitter_boundary"] <= jitter_margin
            if np.isfinite(jitter_margin)
            else False
        )
    else:
        out["near_jitter_boundary"] = False

    out["dist_to_occupancy_boundary"] = out["O"].apply(
        lambda x: nearest_boundary_distance(x, bounds.occupancy_boundaries)
    )
    if len(bounds.occupancy_boundaries) > 0:
        occ_values = out["O"].dropna().to_numpy()
        occ_width = np.nan
        if len(occ_values) > 1:
            occ_width = (np.nanmax(occ_values) - np.nanmin(occ_values)) / 3.0
        occ_margin = occ_width * BOUNDARY_MARGIN_FRAC if np.isfinite(occ_width) else np.nan
        out["near_occupancy_boundary"] = (
            out["dist_to_occupancy_boundary"] <= occ_margin
            if np.isfinite(occ_margin)
            else False
        )
    else:
        out["near_occupancy_boundary"] = False

    out["near_any_boundary"] = (
        out["near_spacing_boundary"].fillna(False)
        | out["near_jitter_boundary"].fillna(False)
        | out["near_occupancy_boundary"].fillna(False)
    )

    return out


# ============================================================
# Bootstrap
# ============================================================

def bootstrap_boundaries(calib_df: pd.DataFrame, n_boot: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []

    for b in range(n_boot):
        sample_idx = rng.integers(0, len(calib_df), len(calib_df))
        boot_df = calib_df.iloc[sample_idx].reset_index(drop=True)
        bounds = propose_boundaries(boot_df)

        row = {
            "boot_id": b,
            "n_spacing_modes": len(bounds.spacing_mode_centers),
        }

        for i, v in enumerate(bounds.spacing_mode_centers):
            row[f"spacing_mode_center_{i+1}"] = v
        for i, v in enumerate(bounds.spacing_boundaries):
            row[f"spacing_boundary_{i+1}"] = v
        for i, v in enumerate(bounds.jitter_boundaries):
            row[f"jitter_boundary_{i+1}"] = v
        for i, v in enumerate(bounds.occupancy_boundaries):
            row[f"occupancy_boundary_{i+1}"] = v

        rows.append(row)

    return pd.DataFrame(rows)


def boundary_ci_table(boot_df: pd.DataFrame) -> pd.DataFrame:
    stats = []
    for col in boot_df.columns:
        if col == "boot_id":
            continue
        vals = pd.to_numeric(boot_df[col], errors="coerce").dropna()
        if len(vals) == 0:
            continue
        stats.append(
            {
                "parameter": col,
                "mean": vals.mean(),
                "std": vals.std(ddof=1) if len(vals) > 1 else 0.0,
                "q025": vals.quantile(0.025),
                "q500": vals.quantile(0.500),
                "q975": vals.quantile(0.975),
                "n": len(vals),
            }
        )
    return pd.DataFrame(stats)


# ============================================================
# Holdout metrics
# ============================================================

def holdout_metrics(df_holdout: pd.DataFrame) -> Dict[str, float]:
    out = {}

    # label consistency proxy:
    # if same dominant spacing family exists, we can measure spacing-class concentration
    label_counts = df_holdout["frozen_label"].value_counts(dropna=True)
    if len(label_counts) > 0:
        out["holdout_label_consistency"] = float(label_counts.iloc[0] / label_counts.sum())
    else:
        out["holdout_label_consistency"] = np.nan

    out["near_boundary_fraction"] = float(df_holdout["near_any_boundary"].mean())

    spacing_counts = df_holdout["spacing_code"].value_counts(dropna=True)
    if len(spacing_counts) > 0:
        total = spacing_counts.sum()
        probs = spacing_counts / total
        out["class_imbalance_max_fraction"] = float(probs.max())
        out["class_imbalance_entropy"] = float(-(probs * np.log2(probs)).sum())
    else:
        out["class_imbalance_max_fraction"] = np.nan
        out["class_imbalance_entropy"] = np.nan

    return out


def version_to_version_relabel_rate(df: pd.DataFrame, bounds_a: BoundarySet, bounds_b: BoundarySet) -> float:
    a = apply_frozen_boundaries(df, bounds_a)["frozen_label"]
    b = apply_frozen_boundaries(df, bounds_b)["frozen_label"]
    valid = a.notna() & b.notna()
    if valid.sum() == 0:
        return np.nan
    return float((a[valid] != b[valid]).mean())


# ============================================================
# Main
# ============================================================

def main() -> None:
    np.random.seed(RANDOM_SEED)
    ensure_outdir(OUTDIR)

    # -------------------------
    # Load and merge
    # -------------------------
    results_df = load_results(RESULT_JSON)
    ann_df = load_annotations(ANNOTATION_CSV)
    df = merge_results_annotations(results_df, ann_df)

    # Keep only positive set for calibration/holdout boundary experiment
    pos_df = df[df["annotation_label"] == "positive"].copy()

    # You can widen this later if needed:
    # pos_df = df[df["annotation_label"].isin(["positive", "ambiguous"])].copy()

    # Basic finite filter for spacing
    pos_df = pos_df[np.isfinite(pos_df["delta_f_hat_mhz"])].copy()

    if len(pos_df) < 6:
        raise ValueError(
            f"Too few positive finite-spacing samples for boundary calibration: n={len(pos_df)}"
        )

    # -------------------------
    # Split calibration / holdout
    # -------------------------
    pos_df = pos_df.sample(frac=1.0, random_state=RANDOM_SEED).reset_index(drop=True)
    n_cal = max(3, int(len(pos_df) * CALIBRATION_FRACTION))
    calib_df = pos_df.iloc[:n_cal].copy()
    holdout_df = pos_df.iloc[n_cal:].copy()

    # -------------------------
    # Step 1. Calibration split
    # -------------------------
    bounds = propose_boundaries(calib_df)

    with open(os.path.join(OUTDIR, "frozen_boundaries.json"), "w", encoding="utf-8") as f:
        json.dump(bounds.to_dict(), f, indent=2)

    # -------------------------
    # Step 2. Holdout split
    # -------------------------
    holdout_labeled = apply_frozen_boundaries(holdout_df, bounds)
    holdout_labeled = compute_near_boundary_flags(holdout_labeled, bounds)
    holdout_labeled.to_csv(os.path.join(OUTDIR, "holdout_labeled.csv"), index=False)

    hold_metrics = holdout_metrics(holdout_labeled)
    hold_metrics_df = pd.DataFrame([hold_metrics])
    hold_metrics_df.to_csv(os.path.join(OUTDIR, "holdout_metrics.csv"), index=False)

    # -------------------------
    # Step 3. Bootstrap
    # -------------------------
    boot_df = bootstrap_boundaries(calib_df, n_boot=N_BOOT, seed=RANDOM_SEED)
    boot_df.to_csv(os.path.join(OUTDIR, "bootstrap_boundaries.csv"), index=False)

    ci_df = boundary_ci_table(boot_df)
    ci_df.to_csv(os.path.join(OUTDIR, "boundary_ci_table.csv"), index=False)

    # -------------------------
    # Version-to-version relabel rate
    # Compare original frozen bounds vs perturbed bootstrap median bounds
    # -------------------------
    # construct a "v2" boundary set from bootstrap medians
    def median_or_empty(prefix: str, nmax: int = 10) -> List[float]:
        out_vals = []
        for i in range(1, nmax + 1):
            col = f"{prefix}_{i}"
            if col in boot_df.columns:
                vals = pd.to_numeric(boot_df[col], errors="coerce").dropna()
                if len(vals) > 0:
                    out_vals.append(float(vals.median()))
        return out_vals

    bounds_v2 = BoundarySet(
        spacing_mode_centers=median_or_empty("spacing_mode_center"),
        spacing_boundaries=median_or_empty("spacing_boundary"),
        jitter_boundaries=median_or_empty("jitter_boundary"),
        occupancy_boundaries=median_or_empty("occupancy_boundary"),
        dwell_boundaries=[],
    )

    relabel_rate = version_to_version_relabel_rate(pos_df, bounds, bounds_v2)
    pd.DataFrame(
        [{"relabel_rate_v1_to_v2": relabel_rate}]
    ).to_csv(os.path.join(OUTDIR, "version_relabel_rate.csv"), index=False)

    # -------------------------
    # Human-readable summary
    # -------------------------
    summary_lines = []
    summary_lines.append("Experiment 3: Boundary calibration and stability")
    summary_lines.append("=" * 60)
    summary_lines.append(f"n_positive_total = {len(pos_df)}")
    summary_lines.append(f"n_calibration = {len(calib_df)}")
    summary_lines.append(f"n_holdout = {len(holdout_df)}")
    summary_lines.append("")
    summary_lines.append("Frozen boundaries proposed from calibration split:")
    summary_lines.append(json.dumps(bounds.to_dict(), indent=2))
    summary_lines.append("")
    summary_lines.append("Holdout metrics:")
    for k, v in hold_metrics.items():
        summary_lines.append(f"  {k}: {v}")
    summary_lines.append("")
    summary_lines.append(f"Version-to-version relabel rate (v1 vs bootstrap-median v2): {relabel_rate}")

    with open(os.path.join(OUTDIR, "summary.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(summary_lines))

    # -------------------------
    # Console print
    # -------------------------
    print("\n[Calibration boundaries]")
    print(json.dumps(bounds.to_dict(), indent=2))

    print("\n[Holdout metrics]")
    print(hold_metrics_df.to_string(index=False))

    print("\n[Boundary CI table]")
    print(ci_df.to_string(index=False))

    print("\n[Version-to-version relabel rate]")
    print(relabel_rate)

    print(f"\nSaved outputs to: {OUTDIR}")


if __name__ == "__main__":
    main()