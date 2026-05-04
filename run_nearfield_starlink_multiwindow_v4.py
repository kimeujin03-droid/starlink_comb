#!/usr/bin/env python3
"""
run_nearfield_starlink_on_background.py
======================================
Paper-grade scaffold for injecting a TLE-constrained LEO/Starlink-like moving
emitter into an EXTERNALLY generated HERA/pyuvsim/hera_sim background visibility.

Design principles
-----------------
1. This script DOES NOT synthesize a foreground internally.
   It requires a background visibility product from hera_sim / pyuvsim / matvis / real UVH5.

2. This script DOES NOT implement redundant calibration.
   It applies only explicitly named weighted delay / fringe-rate operations.
   If calibration residuals are desired, provide them externally or add a clearly named
   gain-perturbation module outside the main paper-grade path.

3. Starlink/UEMR emission is NOT claimed to be an exact proprietary waveform.
   For paper-grade mode, provide a measured/literature-derived spectral template CSV.
   Controlled hypothesis families are allowed only with an explicit config flag and are
   marked as such in the report.

4. LEO delay is computed using near-field antenna-to-satellite range differences:
       tau_ij(t) = (|r_sat(t)-r_j| - |r_sat(t)-r_i|) / c
   not the plane-wave b dot s / c approximation.

5. Delay-domain metrics use tapering/weights, never a bare rectangular FFT by default.
   Flags are kept as weights. Flagged values are NOT replaced by local medians.

Required background NPZ fields, preferred
-----------------------------------------
    vis_tf              complex array, shape (Ntime, Nfreq)
    freqs_hz            float array, shape (Nfreq,)
    times_jd            float array, shape (Ntime,)
    baseline_enu_m      float array, shape (3,), ant2 - ant1 in ENU meters

Optional NPZ fields
-------------------
    weights_tf          float array, shape (Ntime, Nfreq), 0..1
    flags_tf            bool array, shape (Ntime, Nfreq); converted to weights=~flags
    ant1_enu_m          float array, shape (3,); if absent uses -baseline/2
    ant2_enu_m          float array, shape (3,); if absent uses +baseline/2
    eor_tf              complex array, optional, for subspace comparison

UVH5 loading
------------
UVH5 is supported only if pyuvdata and the project's starlink_uemr helper functions are
available. For reliable near-field geometry, NPZ with baseline_enu_m is recommended.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd
import yaml

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    from skyfield.api import load as skyfield_load, wgs84 as skyfield_wgs84
except Exception as exc:  # pragma: no cover
    skyfield_load = None
    skyfield_wgs84 = None
    SKYFIELD_ERROR = repr(exc)
else:
    SKYFIELD_ERROR = ""

try:
    from scipy.signal.windows import blackmanharris, dpss
except Exception as exc:  # pragma: no cover
    blackmanharris = None
    dpss = None
    SCIPY_ERROR = repr(exc)
else:
    SCIPY_ERROR = ""

try:
    from hera_sim.beams import PolyBeam
except Exception as exc:  # pragma: no cover
    PolyBeam = None
    POLYBEAM_ERROR = repr(exc)
else:
    POLYBEAM_ERROR = ""

C_M_PER_S = 299_792_458.0


@dataclass
class BackgroundContext:
    vis_tf: np.ndarray
    freqs_hz: np.ndarray
    times_jd: np.ndarray
    baseline_enu_m: np.ndarray
    weights_tf: np.ndarray
    ant1_enu_m: np.ndarray
    ant2_enu_m: np.ndarray
    eor_tf: Optional[np.ndarray]
    source_format: str
    source_path: str
    metadata: Dict[str, Any]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Near-field TLE Starlink/UEMR injection on external HERA background visibility."
    )
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--background", type=Path, default=None, help="NPZ or UVH5 background visibility. Overrides config.")
    p.add_argument("--tle-path", type=Path, default=None, help="TLE catalog. Overrides config.")
    p.add_argument("--output-dir", type=Path, default=None)
    p.add_argument("--start-window", type=int, default=None, help="Override multi_window.start_window.")
    p.add_argument("--max-windows", type=int, default=None, help="Override multi_window.max_windows.")
    p.add_argument("--max-sats-per-window", type=int, default=None, help="Override multi_window.max_sats_per_window.")
    p.add_argument("--max-scan-satellites", type=int, default=None, help="Override multi_window.max_scan_satellites.")
    p.add_argument("--peak-alt-min-deg", type=float, default=None, help="Override multi_window.peak_alt_min_deg.")
    p.add_argument("--peak-alt-max-deg", type=float, default=None, help="Override multi_window.peak_alt_max_deg.")
    p.add_argument("--cap-selection-method", default=None, help="Override multi_window.cap_selection_method.")
    p.add_argument("--pol", default=None, help="Override uvh5_selection.pol / selected output polarization.")
    p.add_argument("--polarization-mode", default=None, help="Override starlink.polarization.mode.")
    p.add_argument("--polarization-contrast", type=float, default=None, help="Override starlink.polarization.contrast.")
    p.add_argument("--save-window-arrays", action="store_true", help="Save raw/processed multi-window arrays for waterfall plots.")
    p.add_argument("--phase-null-trials", type=int, default=None, help="Override phase-randomized null trial count.")
    return p.parse_args()


def read_yaml(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _as_bool_flags_or_weights(npz) -> np.ndarray | None:
    if "weights_tf" in npz:
        return np.asarray(npz["weights_tf"], dtype=float)
    if "flags_tf" in npz:
        return (~np.asarray(npz["flags_tf"], dtype=bool)).astype(float)
    return None


def load_background_npz(path: Path) -> BackgroundContext:
    data = np.load(path, allow_pickle=True)
    required = ["vis_tf", "freqs_hz", "times_jd", "baseline_enu_m"]
    missing = [k for k in required if k not in data]
    if missing:
        raise ValueError(f"Background NPZ missing required fields: {missing}")

    vis_tf = np.asarray(data["vis_tf"])
    freqs_hz = np.asarray(data["freqs_hz"], dtype=float)
    times_jd = np.asarray(data["times_jd"], dtype=float)
    baseline = np.asarray(data["baseline_enu_m"], dtype=float).reshape(3)

    if vis_tf.shape != (len(times_jd), len(freqs_hz)):
        raise ValueError(
            f"vis_tf shape {vis_tf.shape} does not match times/freqs "
            f"({len(times_jd)}, {len(freqs_hz)})"
        )

    weights = _as_bool_flags_or_weights(data)
    if weights is None:
        weights = np.ones(vis_tf.shape, dtype=float)
    if weights.shape != vis_tf.shape:
        raise ValueError(f"weights/flags shape {weights.shape} does not match vis_tf {vis_tf.shape}")
    weights = np.clip(weights.astype(float), 0.0, 1.0)

    if "ant1_enu_m" in data and "ant2_enu_m" in data:
        ant1 = np.asarray(data["ant1_enu_m"], dtype=float).reshape(3)
        ant2 = np.asarray(data["ant2_enu_m"], dtype=float).reshape(3)
    else:
        # Single-baseline near-field model about the baseline midpoint.
        ant1 = -0.5 * baseline
        ant2 = +0.5 * baseline

    eor_tf = np.asarray(data["eor_tf"]) if "eor_tf" in data else None

    md = {}
    for k in data.files:
        if k not in {"vis_tf", "freqs_hz", "times_jd", "baseline_enu_m", "weights_tf", "flags_tf", "ant1_enu_m", "ant2_enu_m", "eor_tf"}:
            try:
                md[k] = data[k].tolist()
            except Exception:
                md[k] = str(data[k])

    return BackgroundContext(
        vis_tf=vis_tf.astype(complex),
        freqs_hz=freqs_hz,
        times_jd=times_jd,
        baseline_enu_m=baseline,
        weights_tf=weights,
        ant1_enu_m=ant1,
        ant2_enu_m=ant2,
        eor_tf=eor_tf,
        source_format="npz",
        source_path=str(path),
        metadata=md,
    )


def load_background_uvh5(path: Path, cfg: Dict[str, Any]) -> BackgroundContext:
    """Best-effort UVH5 loader using project helpers.

    For true near-field modeling, this loader still represents antennas as +/- baseline/2
    unless the project helper returns ENU antenna positions. NPZ is preferred.
    """
    try:
        from starlink_uemr.mitigation.hera_like_filters import (  # type: ignore
            read_uvh5,
            choose_antpairpol,
            get_freqs_hz,
            get_times_jd,
            get_vis,
            baseline_metadata,
        )
    except Exception as exc:
        raise ImportError(
            "UVH5 loading requires starlink_uemr.mitigation.hera_like_filters. "
            "Use a background NPZ instead."
        ) from exc

    sel = cfg.get("uvh5_selection", {})
    ant1 = sel.get("ant1", None)
    ant2 = sel.get("ant2", None)
    pol = sel.get("pol", None)
    uvd = read_uvh5(path)
    key = choose_antpairpol(uvd, ant1, ant2, pol)
    freqs_hz = get_freqs_hz(uvd)
    times_jd = get_times_jd(uvd, key)
    vis_tf = get_vis(uvd, key)
    meta = baseline_metadata(uvd, key[0], key[1])
    baseline = np.asarray(meta["baseline_enu_m"], dtype=float).reshape(3)

    weights = np.ones_like(np.abs(vis_tf), dtype=float)
    # If a project-specific flag extractor exists, user should export NPZ instead.
    return BackgroundContext(
        vis_tf=np.asarray(vis_tf, dtype=complex),
        freqs_hz=np.asarray(freqs_hz, dtype=float),
        times_jd=np.asarray(times_jd, dtype=float),
        baseline_enu_m=baseline,
        weights_tf=weights,
        ant1_enu_m=-0.5 * baseline,
        ant2_enu_m=+0.5 * baseline,
        eor_tf=None,
        source_format="uvh5_via_project_helpers",
        source_path=str(path),
        metadata={"selected_antpairpol": list(key), **meta},
    )


def load_background(path: Path, cfg: Dict[str, Any]) -> BackgroundContext:
    suffix = path.suffix.lower()
    if suffix == ".npz":
        return load_background_npz(path)
    if suffix == ".uvh5":
        return load_background_uvh5(path, cfg)
    raise ValueError(f"Unsupported background format: {path}")


def select_satellite_from_tle(tle_path: Path, cfg: Dict[str, Any], times_jd: np.ndarray, site_cfg: Dict[str, Any]):
    if skyfield_load is None:
        raise ImportError(f"skyfield unavailable: {SKYFIELD_ERROR}")
    sats = skyfield_load.tle_file(str(tle_path))
    if not sats:
        raise ValueError(f"No satellites in TLE file: {tle_path}")

    sat_cfg = cfg.get("starlink", {})
    name = sat_cfg.get("satellite_name")
    if name:
        for s in sats:
            if s.name == name or name.upper() in (s.name or "").upper():
                return s, {"satellite_name": s.name, "selection": "name"}
        raise ValueError(f"Satellite not found: {name}")

    # Scan candidates over the provided time window or explicit window around times_jd.
    ts = skyfield_load.timescale()
    lat = float(site_cfg["lat_deg"])
    lon = float(site_cfg["lon_deg"])
    elev_m = float(site_cfg.get("elev_m", 0.0))
    observer = skyfield_wgs84.latlon(latitude_degrees=lat, longitude_degrees=lon, elevation_m=elev_m)

    peak_alt_min = float(sat_cfg.get("peak_alt_min_deg", 35.0))
    peak_alt_max = float(sat_cfg.get("peak_alt_max_deg", 65.0))
    max_scan = int(sat_cfg.get("max_scan_satellites", 2000))

    t = ts.tt_jd(times_jd)
    rows = []
    for idx, sat in enumerate(sats[:max_scan]):
        try:
            alt = (sat - observer).at(t).altaz()[0].degrees
            imax = int(np.nanargmax(alt))
            peak = float(alt[imax])
            if peak_alt_min <= peak <= peak_alt_max:
                rows.append((abs(peak - 0.5 * (peak_alt_min + peak_alt_max)), idx, sat, peak, float(times_jd[imax])))
        except Exception:
            continue
    if not rows:
        # Fall back to best visible, but mark it.
        best = None
        for idx, sat in enumerate(sats[:max_scan]):
            try:
                alt = (sat - observer).at(t).altaz()[0].degrees
                peak = float(np.nanmax(alt))
                if best is None or peak > best[0]:
                    best = (peak, idx, sat, float(times_jd[int(np.nanargmax(alt))]))
            except Exception:
                continue
        if best is None:
            raise ValueError("No usable satellite found in TLE scan.")
        peak, idx, sat, peak_jd = best
        return sat, {
            "satellite_name": sat.name,
            "satellite_index": idx,
            "selection": "fallback_best_peak_alt_outside_requested_range",
            "peak_alt_deg": peak,
            "peak_jd": peak_jd,
        }

    rows.sort(key=lambda x: x[0])
    _, idx, sat, peak, peak_jd = rows[0]
    return sat, {
        "satellite_name": sat.name,
        "satellite_index": idx,
        "selection": "peak_altitude_range_scan",
        "peak_alt_deg": peak,
        "peak_jd": peak_jd,
        "peak_altitude_requested_range_deg": [peak_alt_min, peak_alt_max],
    }


def altaz_to_enu_m(alt_deg: np.ndarray, az_deg: np.ndarray, distance_km: np.ndarray) -> np.ndarray:
    alt = np.deg2rad(alt_deg)
    az = np.deg2rad(az_deg)
    r_m = distance_km * 1e3
    east = r_m * np.cos(alt) * np.sin(az)
    north = r_m * np.cos(alt) * np.cos(az)
    up = r_m * np.sin(alt)
    return np.column_stack([east, north, up])


def compute_track_and_nearfield(sat, times_jd: np.ndarray, site_cfg: Dict[str, Any], ant1_enu: np.ndarray, ant2_enu: np.ndarray, freqs_hz: np.ndarray, dt_sec: float, channel_width_hz: float) -> Tuple[pd.DataFrame, Dict[str, np.ndarray]]:
    ts = skyfield_load.timescale()
    observer = skyfield_wgs84.latlon(
        latitude_degrees=float(site_cfg["lat_deg"]),
        longitude_degrees=float(site_cfg["lon_deg"]),
        elevation_m=float(site_cfg.get("elev_m", 0.0)),
    )
    t = ts.tt_jd(times_jd)
    difference = sat - observer
    app = difference.at(t)
    alt, az, distance = app.altaz()
    alt_deg = np.asarray(alt.degrees, dtype=float)
    az_deg = np.asarray(az.degrees, dtype=float)
    range_km = np.asarray(distance.km, dtype=float)
    sat_enu = altaz_to_enu_m(alt_deg, az_deg, range_km)

    r1 = np.linalg.norm(sat_enu - ant1_enu[None, :], axis=1)
    r2 = np.linalg.norm(sat_enu - ant2_enu[None, :], axis=1)
    tau_s = (r2 - r1) / C_M_PER_S

    time_sec = (times_jd - times_jd[0]) * 86400.0
    if len(time_sec) > 1:
        tau_dot = np.gradient(tau_s, time_sec)
        range_rate_m_s = np.gradient(range_km * 1e3, time_sec)
    else:
        tau_dot = np.zeros_like(tau_s)
        range_rate_m_s = np.zeros_like(tau_s)

    fringe_rate_hz = tau_dot[:, None] * freqs_hz[None, :]
    doppler_hz = -range_rate_m_s[:, None] / C_M_PER_S * freqs_hz[None, :]
    sinc_time = np.sinc(fringe_rate_hz * dt_sec)
    sinc_freq = np.sinc(tau_s[:, None] * channel_width_hz)
    attenuation = np.abs(sinc_time * sinc_freq)

    df = pd.DataFrame({
        "jd": times_jd,
        "time_sec": time_sec,
        "alt_deg": alt_deg,
        "az_deg": az_deg,
        "range_km": range_km,
        "range_rate_m_s": range_rate_m_s,
        "tau_s": tau_s,
        "tau_dot_s_per_s": tau_dot,
    })
    comp = {
        "sat_enu_m": sat_enu,
        "tau_s": tau_s,
        "tau_dot_s_per_s": tau_dot,
        "fringe_rate_hz": fringe_rate_hz,
        "doppler_hz": doppler_hz,
        "sinc_time": sinc_time,
        "sinc_freq": sinc_freq,
        "attenuation": attenuation,
    }
    return df, comp


def evaluate_beam(track: pd.DataFrame, freqs_hz: np.ndarray, mode: str) -> Tuple[np.ndarray, str, Dict[str, Any]]:
    if mode == "none":
        return np.ones((len(track), len(freqs_hz)), dtype=float), "none", {}
    if mode != "polybeam_fagnoni19":
        raise ValueError(f"Unsupported beam mode: {mode}")
    if PolyBeam is None:
        raise ImportError(
            "hera_sim PolyBeam is required for paper-grade beam mode. "
            f"Import error: {POLYBEAM_ERROR}"
        )
    beam = PolyBeam.like_fagnoni19()
    az = np.deg2rad(track["az_deg"].to_numpy())
    za = np.pi / 2.0 - np.deg2rad(track["alt_deg"].to_numpy())
    ef = np.asarray(beam.efield_eval(az_array=az, za_array=za, freq_array=freqs_hz))
    power = np.abs(ef) ** 2
    shape = power.shape
    nt, nf = len(track), len(freqs_hz)
    t_axes = [i for i, s in enumerate(shape) if s == nt]
    f_axes = [i for i, s in enumerate(shape) if s == nf]
    if not t_axes or not f_axes:
        raise ValueError(f"Cannot infer PolyBeam axes from shape {shape}; nt={nt}, nf={nf}")
    out = np.moveaxis(power, [t_axes[-1], f_axes[0]], [0, 1])
    if out.ndim > 2:
        out = out.mean(axis=tuple(range(2, out.ndim)))
    out = np.asarray(out, dtype=float)
    raw_max = float(np.nanmax(out)) if out.size else np.nan
    raw_min = float(np.nanmin(out)) if out.size else np.nan
    out = np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)
    # PolyBeam is a normalized primary-beam approximation, but its Chebyshev
    # polynomial can produce unphysical excursions outside the fitted domain.
    # Use a bounded response for injected flux and retain raw extrema in meta.
    out = np.clip(out, 0.0, 1.0)
    return np.asarray(out, dtype=float), "hera_sim.PolyBeam.like_fagnoni19", {
        "efield_shape": list(shape),
        "raw_power_min": raw_min,
        "raw_power_max": raw_max,
        "applied_power_min": float(np.nanmin(out)) if out.size else np.nan,
        "applied_power_max": float(np.nanmax(out)) if out.size else np.nan,
        "clipped_to_unit_power": bool(np.isfinite(raw_max) and raw_max > 1.0),
    }


def load_spectral_template(path: Path, freqs_hz: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
    df = pd.read_csv(path)
    if "freq_hz" in df.columns:
        f = df["freq_hz"].to_numpy(dtype=float)
    elif "freq_mhz" in df.columns:
        f = df["freq_mhz"].to_numpy(dtype=float) * 1e6
    else:
        raise ValueError("Spectral template CSV must contain freq_hz or freq_mhz")

    if "relative_power" in df.columns:
        y = df["relative_power"].to_numpy(dtype=float)
        meaning = "relative_power"
    elif "relative_amplitude" in df.columns:
        y = df["relative_amplitude"].to_numpy(dtype=float)
        meaning = "relative_amplitude"
    elif "flux_jy" in df.columns:
        y = df["flux_jy"].to_numpy(dtype=float)
        meaning = "flux_jy"
    else:
        raise ValueError("Spectral template CSV must contain relative_power, relative_amplitude, or flux_jy")

    order = np.argsort(f)
    f, y = f[order], y[order]
    interp = np.interp(freqs_hz, f, y, left=0.0, right=0.0)
    interp = np.clip(interp, 0.0, None)
    if np.nanmax(interp) <= 0:
        raise ValueError("Spectral template is zero over the requested frequency range.")
    if meaning in {"relative_power", "flux_jy"}:
        amp_shape = interp / np.nanmax(interp)
    else:
        amp_shape = interp / np.nanmax(interp)
    return amp_shape.astype(float), {
        "template_path": str(path),
        "template_columns": list(df.columns),
        "template_meaning": meaning,
        "nonzero_fraction_on_grid": float(np.mean(amp_shape > 0)),
    }


def load_time_template(path: Optional[Path], times_jd: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
    if path is None:
        return np.ones(len(times_jd), dtype=float), {"time_template": "constant_unity"}
    df = pd.read_csv(path)
    if "jd" in df.columns:
        x = df["jd"].to_numpy(dtype=float)
        xp = times_jd
    elif "time_sec" in df.columns:
        x = df["time_sec"].to_numpy(dtype=float)
        xp = (times_jd - times_jd[0]) * 86400.0
    else:
        raise ValueError("Time template CSV must contain jd or time_sec")
    col = "duty" if "duty" in df.columns else "relative_amplitude" if "relative_amplitude" in df.columns else None
    if col is None:
        raise ValueError("Time template CSV must contain duty or relative_amplitude")
    y = np.clip(df[col].to_numpy(dtype=float), 0.0, None)
    order = np.argsort(x)
    duty = np.interp(xp, x[order], y[order], left=0.0, right=0.0)
    if np.nanmax(duty) > 0:
        duty = duty / np.nanmax(duty)
    return duty.astype(float), {"time_template_path": str(path), "time_template_column": col}


def _gaussian_line_profile(freqs_hz: np.ndarray, center_hz: float, width_hz: float) -> np.ndarray:
    width = max(float(width_hz), 1.0)
    return np.exp(-0.5 * ((freqs_hz - center_hz) / width) ** 2)


def _smooth_top_hat(freqs_hz: np.ndarray, center_hz: float, width_hz: float, edge_hz: float) -> np.ndarray:
    """Differentiable rectangular band used for literature-measured broadband windows."""
    half = 0.5 * float(width_hz)
    edge = max(float(edge_hz), 1.0)
    x = np.abs(freqs_hz - center_hz)
    return 1.0 / (1.0 + np.exp((x - half) / edge))


def literature_uemr_spectrum(freqs_hz: np.ndarray, cfg: Dict[str, Any], channel_width_hz: float) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Build a literature-anchored Starlink UEMR amplitude template.

    This is not a proprietary Starlink waveform reconstruction. It encodes measured
    LOFAR low-frequency UEMR facts as a parameterized observation template:

    - Di Vruno et al. 2023: Gen-1 Starlink-associated UEMR between 110--188 MHz,
      broadband features plus narrowband lines at 125, 135, 143.05, 150, 175 MHz;
      narrowband bandwidth <12.2 kHz; broad-band flux densities 0.1--10 Jy and
      some narrowband features 10--500 Jy.
    - Bassa et al. 2024: v2-Mini/v2-Mini DTC broadband emission over 40--70 MHz and
      110--188 MHz; reported spectral flux density ranges include 15--1300 Jy in
      56--66 MHz and 2--100 Jy in 8 MHz windows centered near 120 and 161 MHz;
      periodic features in 157--165 MHz with 48.8, 65, and 97.5 kHz spacing.

    The output is normalized to max=1 and later scaled by reference_flux_jy.
    """
    em = cfg.get("starlink", {}).get("emission_model", {})
    model = str(em.get("literature_model", "bassa2024_v2mini_lofar")).lower()
    rng = np.random.default_rng(int(cfg.get("experiment", {}).get("random_seed", 0)))
    span = max(float(np.ptp(freqs_hz)), 1.0)
    spec_flux = np.zeros_like(freqs_hz, dtype=float)
    components: list[dict[str, Any]] = []

    def add_band(label: str, center_mhz: float, width_mhz: float, flux_jy: float, edge_mhz: float = 0.5):
        prof = _smooth_top_hat(freqs_hz, center_mhz * 1e6, width_mhz * 1e6, edge_mhz * 1e6)
        spec_flux[:] += float(flux_jy) * prof
        components.append({"type": "broadband_window", "label": label, "center_mhz": center_mhz, "width_mhz": width_mhz, "flux_jy": float(flux_jy)})

    def add_line(label: str, center_mhz: float, peak_flux_jy: float, intrinsic_width_hz: float = 12_200.0):
        # At coarse HERA-like channelization, a sub-channel line is diluted when
        # interpreted as channel-averaged flux density. This prevents unrealistically
        # strong 12 kHz lines in 122 kHz channels.
        dilute = bool(em.get("dilute_subchannel_lines", True))
        effective_peak = float(peak_flux_jy)
        if dilute:
            effective_peak *= min(1.0, float(intrinsic_width_hz) / max(float(channel_width_hz), 1.0))
        # Use max(width, half-channel) so the line is visible on the simulation grid
        # as a channel-averaged narrow feature, not as an unresolved delta spike.
        render_width_hz = max(float(intrinsic_width_hz), 0.5 * float(channel_width_hz))
        prof = _gaussian_line_profile(freqs_hz, center_mhz * 1e6, render_width_hz)
        spec_flux[:] += effective_peak * prof
        components.append({
            "type": "narrowband_line", "label": label, "center_mhz": center_mhz,
            "intrinsic_width_hz": intrinsic_width_hz, "input_peak_flux_jy": float(peak_flux_jy),
            "channel_averaged_peak_flux_jy": float(effective_peak),
            "dilute_subchannel_lines": dilute,
        })

    def add_comb(label: str, start_mhz: float, stop_mhz: float, spacing_khz: float, peak_flux_jy: float, intrinsic_width_hz: float = 8_000.0):
        centers_hz = np.arange(start_mhz * 1e6, stop_mhz * 1e6 + spacing_khz * 1e3, spacing_khz * 1e3)
        # Avoid thousands of tiny lines if someone sets too fine a spacing accidentally.
        max_teeth = int(em.get("max_comb_teeth", 512))
        if len(centers_hz) > max_teeth:
            centers_hz = centers_hz[np.linspace(0, len(centers_hz) - 1, max_teeth).astype(int)]
        dilute = bool(em.get("dilute_subchannel_lines", True))
        effective_peak = float(peak_flux_jy) * (min(1.0, float(intrinsic_width_hz) / max(float(channel_width_hz), 1.0)) if dilute else 1.0)
        render_width_hz = max(float(intrinsic_width_hz), 0.35 * float(channel_width_hz))
        envelope = _smooth_top_hat(freqs_hz, 0.5 * (start_mhz + stop_mhz) * 1e6, (stop_mhz - start_mhz) * 1e6, 0.25e6)
        comb = np.zeros_like(freqs_hz, dtype=float)
        for c in centers_hz:
            # Low-amplitude tooth-to-tooth modulation: measurement-inspired, not waveform claim.
            amp = effective_peak * rng.uniform(0.55, 1.0)
            comb += amp * _gaussian_line_profile(freqs_hz, c, render_width_hz)
        spec_flux[:] += comb * envelope
        components.append({
            "type": "comb", "label": label, "start_mhz": start_mhz, "stop_mhz": stop_mhz,
            "spacing_khz": spacing_khz, "n_teeth_rendered": int(len(centers_hz)),
            "input_peak_flux_jy": float(peak_flux_jy), "channel_averaged_peak_flux_jy": float(effective_peak),
        })

    # User may override representative flux values while retaining literature positions.
    if "di_vruno" in model or model in {"gen1", "hybrid_gen1_v2mini"}:
        broad_flux = float(em.get("di_vruno_broadband_flux_jy", 3.0))  # inside 0.1--10 Jy range
        add_band("DiVruno2023 broad 110-188 MHz", 149.0, 78.0, broad_flux, edge_mhz=1.0)
        line_flux = float(em.get("di_vruno_narrowband_peak_flux_jy", 100.0))  # inside 10--500 Jy range
        for mhz in em.get("di_vruno_narrowband_lines_mhz", [125.0, 135.0, 143.05, 150.0, 175.0]):
            add_line("DiVruno2023 narrowband", float(mhz), line_flux, intrinsic_width_hz=float(em.get("di_vruno_line_width_hz", 12_200.0)))

    if "bassa" in model or "v2" in model or model == "hybrid_gen1_v2mini":
        # Only windows overlapping the requested freq grid contribute. 56--66 MHz is kept
        # for completeness if the user simulates low-frequency LOFAR-like bands.
        low_flux = float(em.get("bassa_lowband_flux_jy", 200.0))  # 15--1300 Jy reported range
        add_band("Bassa2024 v2-Mini 56-66 MHz", 61.0, 10.0, low_flux, edge_mhz=0.5)
        hba_flux = float(em.get("bassa_hba_window_flux_jy", 30.0))  # 2--100 Jy reported range
        add_band("Bassa2024 v2-Mini 120 MHz 8 MHz", 120.0, 8.0, hba_flux, edge_mhz=0.35)
        add_band("Bassa2024 v2-Mini 161 MHz 8 MHz", 161.0, 8.0, hba_flux, edge_mhz=0.35)
        if bool(em.get("include_bassa_comb_157_165", True)):
            comb_peak = float(em.get("bassa_comb_peak_flux_jy", 25.0))
            for spacing in em.get("bassa_v2mini_comb_spacings_khz", [48.8, 65.0, 97.5]):
                add_comb("Bassa2024 v2-Mini 157-165 MHz comb", 157.0, 165.0, float(spacing), comb_peak, intrinsic_width_hz=float(em.get("bassa_comb_tooth_width_hz", 8_000.0)))
        if bool(em.get("include_v15_comb_157_165", False)):
            add_comb("Bassa2024/Gen1 v1.5 157-165 MHz comb", 157.0, 165.0, 50.0, float(em.get("v15_comb_peak_flux_jy", 25.0)), intrinsic_width_hz=float(em.get("v15_comb_tooth_width_hz", 8_000.0)))

    if np.nanmax(spec_flux) <= 0:
        raise ValueError(f"Literature UEMR template '{model}' has zero support on the requested frequency grid.")
    amp_shape = spec_flux / max(float(np.nanmax(spec_flux)), 1e-30)
    return amp_shape.astype(float), {
        "emission_model_type": "literature_parameterized_uemr_template_not_proprietary_waveform",
        "literature_model": model,
        "normalization": "template normalized to max=1, then scaled by reference_flux_jy and range/beam/smearing",
        "channel_width_hz": float(channel_width_hz),
        "components": components,
        "citations_needed": ["Di Vruno et al. 2023 A&A", "Bassa et al. 2024 A&A"],
    }


def polarization_scaling(ctx: BackgroundContext, cfg: Dict[str, Any], n_time: int) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Return scalar polarization scaling for the selected baseline product.

    This implements measurement-inspired anti-correlated XX/YY amplitude handling when
    only one polarization product is present. It is not a Jones-matrix beam model.
    """
    pol_cfg = cfg.get("starlink", {}).get("polarization", {})
    mode = str(pol_cfg.get("mode", "scalar")).lower()
    selected_pol = str(cfg.get("uvh5_selection", {}).get("pol", ctx.metadata.get("selected_antpairpol", [None, None, "ee"])[-1] if isinstance(ctx.metadata.get("selected_antpairpol"), list) else "ee")).lower()
    t = np.linspace(-1.0, 1.0, n_time)
    if mode in {"scalar", "none"}:
        factor = np.full(n_time, float(pol_cfg.get("scalar_factor", cfg.get("starlink", {}).get("scalar_polarization_factor", 1.0))), dtype=float)
        return factor, {"mode": "scalar", "selected_pol": selected_pol, "jones_model": False}
    if mode != "anti_correlated_xx_yy":
        raise ValueError(f"Unsupported starlink.polarization.mode: {mode}")
    mean = float(pol_cfg.get("mean", 1.0))
    contrast = float(pol_cfg.get("contrast", 0.35))
    phase = float(pol_cfg.get("phase_rad", 0.0))
    variation = np.sin(2.0 * np.pi * (0.5 * t + 0.5) + phase)
    xx = np.clip(mean * (1.0 + contrast * variation), 0.0, None)
    yy = np.clip(mean * (1.0 - contrast * variation), 0.0, None)
    # HERA-style ee/nn are used as approximate XX/YY analogues here.
    if selected_pol in {"xx", "ee", "x", "e"}:
        factor = xx
        mapped = "XX/EE"
    elif selected_pol in {"yy", "nn", "y", "n"}:
        factor = yy
        mapped = "YY/NN"
    else:
        factor = 0.5 * (xx + yy)
        mapped = "unknown_pol_mean_of_anti_correlated_pair"
    return factor.astype(float), {
        "mode": "anti_correlated_xx_yy",
        "selected_pol": selected_pol,
        "mapped_component": mapped,
        "mean": mean,
        "contrast": contrast,
        "jones_model": False,
        "limitation": "Scalar per-pol anti-correlation only; does not replace a full Jones-matrix antenna/satellite polarization model.",
    }


def selected_polarization_product(ctx: BackgroundContext, cfg: Dict[str, Any]) -> str:
    meta_pol = "ee"
    selected = ctx.metadata.get("selected_antpairpol")
    if isinstance(selected, list) and selected:
        meta_pol = str(selected[-1])
    return str(cfg.get("uvh5_selection", {}).get("pol", meta_pol)).lower()


def source_coherency_matrix(S_tf: np.ndarray, cfg: Dict[str, Any]) -> Tuple[np.ndarray, Dict[str, Any]]:
    pol_cfg = cfg.get("starlink", {}).get("polarization", {})
    mode = str(pol_cfg.get("mode", "scalar")).lower()
    nt, nf = S_tf.shape
    C = np.zeros((nt, nf, 2, 2), dtype=complex)
    if mode in {"jones_unpolarized", "unpolarized"}:
        C[..., 0, 0] = S_tf
        C[..., 1, 1] = S_tf
        return C, {
            "mode": "jones_unpolarized",
            "source_coherency_model": "unpolarized",
            "stokes": "I_only",
            "jones_model": True,
        }
    if mode in {"jones_anti_correlated", "anti_correlated_jones", "anti_correlated_xx_yy"}:
        contrast = float(pol_cfg.get("contrast", 0.35))
        phase = float(pol_cfg.get("phase_rad", 0.0))
        t_norm = np.linspace(0.0, 2.0 * np.pi, nt)
        q_t = contrast * np.sin(t_norm + phase)
        q_tf = q_t[:, None] * S_tf
        C[..., 0, 0] = np.clip(S_tf + q_tf, 0.0, None)
        C[..., 1, 1] = np.clip(S_tf - q_tf, 0.0, None)
        return C, {
            "mode": "jones_anti_correlated",
            "source_coherency_model": "anti_correlated",
            "contrast": contrast,
            "phase_rad": phase,
            "reference": "Bassa et al. 2024 A&A LOFAR v2-Mini UEMR anti-correlated XX/YY motivation",
            "jones_model": True,
            "d_terms": "zero",
            "limitation": "Diagonal Jones approximation with parameterized source coherency; true Starlink Stokes Q/U/V is not measured here.",
        }
    raise ValueError(f"Unsupported Jones source coherency mode: {mode}")


def project_diagonal_jones_visibility(
    source_amp_tf: np.ndarray,
    beam_power_tf: np.ndarray,
    phase_tf: np.ndarray,
    ctx: BackgroundContext,
    cfg: Dict[str, Any],
) -> Tuple[np.ndarray, Dict[str, Any], Dict[str, np.ndarray]]:
    C_src, c_meta = source_coherency_matrix(source_amp_tf, cfg)
    selected_pol = selected_polarization_product(ctx, cfg)
    pol_map = {
        "xx": (0, 0), "ee": (0, 0), "x": (0, 0), "e": (0, 0),
        "xy": (0, 1), "en": (0, 1),
        "yx": (1, 0), "ne": (1, 0),
        "yy": (1, 1), "nn": (1, 1), "y": (1, 1), "n": (1, 1),
    }
    if selected_pol not in pol_map:
        raise ValueError(f"Unknown selected polarization product: {selected_pol}")
    ip, jp = pol_map[selected_pol]
    J_power = beam_power_tf
    V_full = C_src * J_power[:, :, None, None]
    vis = V_full[..., ip, jp] * phase_tf
    stokes_i = 0.5 * (V_full[..., 0, 0] + V_full[..., 1, 1])
    stokes_q = 0.5 * (V_full[..., 0, 0] - V_full[..., 1, 1])
    stokes_u = 0.5 * (V_full[..., 0, 1] + V_full[..., 1, 0])
    stokes_v = 0.5 * (V_full[..., 0, 1] - V_full[..., 1, 0]) / 1j
    diagnostics = {
        "power_XX": float(np.nanmean(np.abs(V_full[..., 0, 0]) ** 2)),
        "power_YY": float(np.nanmean(np.abs(V_full[..., 1, 1]) ** 2)),
        "power_XY": float(np.nanmean(np.abs(V_full[..., 0, 1]) ** 2)),
        "power_YX": float(np.nanmean(np.abs(V_full[..., 1, 0]) ** 2)),
        "stokes_I_mean_abs": float(np.nanmean(np.abs(stokes_i))),
        "stokes_Q_mean_abs": float(np.nanmean(np.abs(stokes_q))),
        "stokes_Q_over_I_mean_abs": float(np.nanmean(np.abs(stokes_q)) / max(np.nanmean(np.abs(stokes_i)), 1e-30)),
        "selected_pol_peak_abs_pre_phase": float(np.nanmax(np.abs(V_full[..., ip, jp]))),
    }
    meta = {
        **c_meta,
        "selected_pol": selected_pol,
        "mapped_matrix_element": [int(ip), int(jp)],
        "measurement_equation": "V_ij = J_i C_src J_j^H with diagonal, identical antenna Jones; beam_power_tf = |J|^2.",
        "diagnostics": diagnostics,
        "non_claims": [
            "No proprietary Starlink waveform reconstruction is claimed.",
            "Source coherency C_src is literature-parameterized, not a direct full-Stokes satellite measurement.",
            "D-terms, mutual coupling, and ionospheric Faraday rotation are not modeled.",
        ],
    }
    arrays = {
        "jones_stokes_I_tf": stokes_i,
        "jones_stokes_Q_tf": stokes_q,
        "jones_stokes_U_tf": stokes_u,
        "jones_stokes_V_tf": stokes_v,
    }
    return vis.astype(complex), meta, arrays


def controlled_hypothesis_spectrum(freqs_hz: np.ndarray, cfg: Dict[str, Any], rng: np.random.Generator) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Explicitly labeled non-paper-grade default unless allowed."""
    em = cfg.get("starlink", {}).get("emission_model", {})
    if not bool(em.get("allow_controlled_hypothesis", False)):
        raise ValueError(
            "No measured/literature spectral template provided. For paper-grade runs, set "
            "starlink.emission_model.spectral_template_csv. To run a controlled hypothesis family, "
            "set allow_controlled_hypothesis: true and do not claim exact Starlink waveform reconstruction."
        )
    nf = len(freqs_hz)
    f0 = np.median(freqs_hz)
    span = max(np.ptp(freqs_hz), 1.0)
    f_norm = (freqs_hz - f0) / span
    n_lines = int(em.get("n_comb_components", 17))
    centers = np.linspace(-0.45, 0.45, n_lines) + rng.normal(0, 0.018, n_lines)
    widths = rng.uniform(0.004, 0.018, n_lines)
    amps = rng.lognormal(mean=0.0, sigma=0.5, size=n_lines)
    spec = np.zeros(nf, dtype=float)
    for c, w, a in zip(centers, widths, amps):
        spec += a * np.exp(-0.5 * ((f_norm - c) / w) ** 2)
    floor = float(em.get("broadband_floor", 0.03)) * np.exp(-0.5 * (f_norm / 0.40) ** 2)
    spec = spec + floor
    spec = spec / max(np.max(spec), 1e-12)
    return spec, {
        "emission_model_type": "controlled_hypothesis_family_not_exact_starlink",
        "n_comb_components": n_lines,
        "warning": "Use only as controlled stress-test family unless calibrated to measurements.",
    }


def build_starlink_visibility(ctx: BackgroundContext, cfg: Dict[str, Any], out_dir: Path, sat_override=None, selected_override: Optional[Dict[str, Any]] = None, random_seed_offset: int = 0) -> Tuple[np.ndarray, pd.DataFrame, Dict[str, Any], Dict[str, np.ndarray]]:
    rng = np.random.default_rng(int(cfg.get("experiment", {}).get("random_seed", 0)) + int(random_seed_offset))
    site_cfg = cfg["site"]
    sat_cfg = cfg["starlink"]
    tle_path = Path(sat_cfg["tle_path"])
    if sat_override is None:
        sat, selected = select_satellite_from_tle(tle_path, cfg, ctx.times_jd, site_cfg)
    else:
        sat = sat_override
        selected = dict(selected_override or {})
        selected.setdefault("satellite_name", getattr(sat, "name", "unknown"))
        selected.setdefault("selection", "provided_by_multi_window_enumerator")

    dt_sec = float(np.median(np.diff(ctx.times_jd)) * 86400.0) if len(ctx.times_jd) > 1 else float(cfg.get("time_frequency", {}).get("dt_sec", 10.0))
    if "channel_width_hz" in cfg.get("time_frequency", {}):
        dnu = float(cfg["time_frequency"]["channel_width_hz"])
    else:
        dnu = float(np.median(np.diff(ctx.freqs_hz))) if len(ctx.freqs_hz) > 1 else 1.0

    track, geom = compute_track_and_nearfield(
        sat=sat,
        times_jd=ctx.times_jd,
        site_cfg=site_cfg,
        ant1_enu=ctx.ant1_enu_m,
        ant2_enu=ctx.ant2_enu_m,
        freqs_hz=ctx.freqs_hz,
        dt_sec=dt_sec,
        channel_width_hz=dnu,
    )

    beam_mode = cfg.get("beam", {}).get("mode", "polybeam_fagnoni19")
    beam_tf, beam_name, beam_meta = evaluate_beam(track, ctx.freqs_hz, beam_mode)

    em = sat_cfg.get("emission_model", {})
    spectral_template = em.get("spectral_template_csv")
    literature_model = em.get("literature_model")
    if spectral_template:
        spec_amp, spec_meta = load_spectral_template(Path(spectral_template), ctx.freqs_hz)
        emission_type = "measured_or_literature_spectral_template_csv"
    elif literature_model:
        spec_amp, spec_meta = literature_uemr_spectrum(ctx.freqs_hz, cfg, dnu)
        emission_type = spec_meta["emission_model_type"]
    else:
        spec_amp, spec_meta = controlled_hypothesis_spectrum(ctx.freqs_hz, cfg, rng)
        emission_type = spec_meta["emission_model_type"]

    time_template = em.get("time_template_csv")
    duty_t, duty_meta = load_time_template(Path(time_template) if time_template else None, ctx.times_jd)

    flux_ref = float(sat_cfg.get("reference_flux_jy", sat_cfg.get("flux_jy_ref", 300.0)))
    r_ref = float(sat_cfg.get("reference_range_km", 550.0))
    range_mode = sat_cfg.get("range_attenuation_mode", "flux_density_1_over_r2")
    range_km = track["range_km"].to_numpy(dtype=float)
    if range_mode == "flux_density_1_over_r2":
        range_att = (r_ref / np.clip(range_km, 1e-6, None)) ** 2
    elif range_mode == "field_amplitude_1_over_r":
        range_att = (r_ref / np.clip(range_km, 1e-6, None))
    elif range_mode == "none_observed_apparent_flux":
        range_att = np.ones_like(range_km)
    else:
        raise ValueError(f"Unsupported range_attenuation_mode: {range_mode}")

    spectral_index = float(sat_cfg.get("spectral_index", 0.0))
    freq_ref = float(np.median(ctx.freqs_hz))
    spectral_slope = (ctx.freqs_hz / freq_ref) ** spectral_index

    # Optional deterministic phase texture from Doppler-induced feature drift is not a waveform claim.
    tau = geom["tau_s"]
    phase = np.exp(-2j * np.pi * tau[:, None] * ctx.freqs_hz[None, :])
    attenuation = geom["attenuation"]

    pol_cfg = cfg.get("starlink", {}).get("polarization", {})
    pol_mode = str(pol_cfg.get("mode", "scalar")).lower()
    source_amp_tf = (
        flux_ref
        * range_att[:, None]
        * attenuation
        * duty_t[:, None]
        * spec_amp[None, :]
        * spectral_slope[None, :]
    )
    pol_arrays: Dict[str, np.ndarray] = {}
    if pol_mode.startswith("jones_") or pol_mode in {"unpolarized", "anti_correlated_jones"}:
        vis, pol_meta, pol_arrays = project_diagonal_jones_visibility(
            source_amp_tf=source_amp_tf,
            beam_power_tf=beam_tf,
            phase_tf=phase,
            ctx=ctx,
            cfg=cfg,
        )
        amp_tf = np.abs(vis)
        pol_factor_t = np.ones(len(ctx.times_jd), dtype=float)
    else:
        # Legacy scalar per-pol polarization scaling.
        pol_factor_t, pol_meta = polarization_scaling(ctx, cfg, len(ctx.times_jd))
        amp_tf = source_amp_tf * beam_tf * pol_factor_t[:, None]
        vis = amp_tf * phase
    spectrum_jy = flux_ref * spec_amp * spectral_slope
    sanity = {
        "beam_max": float(np.nanmax(beam_tf)),
        "range_att_max": float(np.nanmax(range_att)),
        "spectrum_jy_max": float(np.nanmax(spectrum_jy)),
        "raw_starlink_peak_abs": float(np.nanmax(np.abs(vis))),
        "raw_background_peak_abs": float(np.nanmax(np.abs(ctx.vis_tf))),
    }
    if sanity["beam_max"] > 10:
        raise ValueError(
            f"Beam response too large for {selected}: beam_max={sanity['beam_max']:.6g}; "
            "check frequency units."
        )
    if sanity["raw_starlink_peak_abs"] > 1e6:
        raise ValueError(
            f"Unphysical Starlink amplitude for {selected}: "
            f"raw_starlink_peak_abs={sanity['raw_starlink_peak_abs']:.6g}; "
            "check flux/range/beam units."
        )

    bright_thr = float(sat_cfg.get("bright_mask_threshold_fraction", 0.03)) * max(np.nanmax(np.abs(vis)), 1e-30)
    loose_thr = float(sat_cfg.get("loose_mask_threshold_fraction", 0.003)) * max(np.nanmax(np.abs(vis)), 1e-30)
    bright_mask = np.abs(vis) >= bright_thr
    loose_mask = np.abs(vis) >= loose_thr

    report = {
        "selected_satellite": selected,
        "near_field_delay": True,
        "tau_definition": "(|r_sat-r_ant2| - |r_sat-r_ant1|) / c",
        "beam_model": beam_name,
        "beam_meta": beam_meta,
        "emission_type": emission_type,
        "spectral_template": spec_meta,
        "time_template": duty_meta,
        "range_attenuation_mode": range_mode,
        "reference_flux_jy": flux_ref,
        "reference_range_km": r_ref,
        "amplitude_sanity": sanity,
        "polarization": pol_meta,
        "polarization_limitation": pol_meta.get("limitation", "Scalar per-pol handling only unless jones_model=true."),
        "dt_sec": dt_sec,
        "channel_width_hz": dnu,
        "peak_alt_deg": float(np.nanmax(track["alt_deg"])),
        "range_min_km": float(np.nanmin(track["range_km"])),
        "range_max_km": float(np.nanmax(track["range_km"])),
        "tau_min_ns": float(np.nanmin(tau) * 1e9),
        "tau_max_ns": float(np.nanmax(tau) * 1e9),
        "max_abs_fringe_rate_mhz": float(np.nanmax(np.abs(geom["fringe_rate_hz"])) * 1e3),
        "max_abs_doppler_hz": float(np.nanmax(np.abs(geom["doppler_hz"]))),
        "peak_abs_jy": float(np.nanmax(np.abs(vis))),
        "mean_abs_jy": float(np.nanmean(np.abs(vis))),
        "median_abs_jy": float(np.nanmedian(np.abs(vis))),
        "bright_mask_fraction": float(np.mean(bright_mask)),
        "loose_mask_fraction": float(np.mean(loose_mask)),
    }
    arrays = {
        "beam_tf": beam_tf,
        "attenuation_tf": attenuation,
        "range_attenuation_t": range_att,
        "spectrum_jy_f": spectrum_jy,
        "spectral_template_f": spec_amp,
        "duty_t": duty_t,
        "polarization_factor_t": pol_factor_t,
        "bright_mask": bright_mask.astype(bool),
        "loose_mask": loose_mask.astype(bool),
        **pol_arrays,
        **geom,
    }
    return vis.astype(complex), track, report, arrays


def make_taper(n: int, method: str, cfg: Dict[str, Any]) -> np.ndarray:
    method = method.lower()
    if method in {"blackman_harris", "blackmanharris"}:
        if blackmanharris is None:
            raise ImportError(f"scipy Blackman-Harris unavailable: {SCIPY_ERROR}")
        return blackmanharris(n, sym=False)
    if method == "hann":
        return np.hanning(n)
    if method == "dpss":
        if dpss is None:
            raise ImportError(f"scipy DPSS unavailable: {SCIPY_ERROR}")
        nw = float(cfg.get("nw", 2.5))
        # First taper only for a conservative single-taper metric.
        return np.asarray(dpss(n, NW=nw, Kmax=1, sym=False)[0], dtype=float)
    raise ValueError(f"Unsupported taper method: {method}")


def weighted_delay_transform(vis_tf: np.ndarray, weights_tf: np.ndarray, taper: np.ndarray) -> np.ndarray:
    w = np.clip(weights_tf, 0.0, 1.0)
    tw = w * taper[None, :]
    # Weighted transform without value replacement. Missing data contributes zero weight.
    x = np.where(np.isfinite(vis_tf), vis_tf, 0.0) * tw
    return np.fft.fftshift(np.fft.fft(x, axis=1), axes=1)


def inverse_weighted_delay_transform(delay_tf: np.ndarray, taper: np.ndarray) -> np.ndarray:
    # Approximate inverse for filtering. We do not divide by taper to avoid exploding edges.
    return np.fft.ifft(np.fft.ifftshift(delay_tf, axes=1), axis=1)


def delay_axis_s(freqs_hz: np.ndarray) -> np.ndarray:
    dnu = float(np.median(np.diff(freqs_hz))) if len(freqs_hz) > 1 else 1.0
    return np.fft.fftshift(np.fft.fftfreq(len(freqs_hz), d=dnu))


def fringe_rate_transform(vis_tf: np.ndarray, times_jd: np.ndarray, weights_tf: np.ndarray, taper_t: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
    nt = len(times_jd)
    dt = float(np.median(np.diff(times_jd)) * 86400.0) if nt > 1 else 1.0
    if taper_t is None:
        taper_t = np.hanning(nt) if nt > 2 else np.ones(nt)
    w_t = np.nanmean(np.clip(weights_tf, 0.0, 1.0), axis=1)
    x = np.where(np.isfinite(vis_tf), vis_tf, 0.0) * (w_t * taper_t)[:, None]
    fr = np.fft.fftshift(np.fft.fftfreq(nt, d=dt))
    return np.fft.fftshift(np.fft.fft(x, axis=0), axes=0), fr


def apply_delay_highpass_weighted(vis_tf: np.ndarray, ctx: BackgroundContext, cfg: Dict[str, Any]) -> Tuple[np.ndarray, Dict[str, Any]]:
    dc = cfg.get("pipeline", {}).get("delay_filter", {})
    if not bool(dc.get("enabled", True)):
        return vis_tf.copy(), {"enabled": False}
    taper_method = dc.get("taper", "blackman_harris")
    taper = make_taper(len(ctx.freqs_hz), taper_method, dc)
    dly = weighted_delay_transform(vis_tf, ctx.weights_tf, taper)
    delays = delay_axis_s(ctx.freqs_hz)

    horizon_ns = np.linalg.norm(ctx.baseline_enu_m) / C_M_PER_S * 1e9
    buffer_ns = float(dc.get("buffer_ns", 100.0))
    cutoff_s = (horizon_ns + buffer_ns) * 1e-9
    # Foreground-avoidance high-pass: remove low delay modes.
    keep = np.abs(delays) >= cutoff_s
    dly_f = dly * keep[None, :]
    out = inverse_weighted_delay_transform(dly_f, taper)
    return out, {
        "enabled": True,
        "taper": taper_method,
        "horizon_ns": float(horizon_ns),
        "buffer_ns": buffer_ns,
        "cutoff_ns": float(cutoff_s * 1e9),
        "kept_delay_fraction": float(np.mean(keep)),
    }


def apply_fringe_filters(vis_tf: np.ndarray, ctx: BackgroundContext, cfg: Dict[str, Any]) -> Tuple[np.ndarray, Dict[str, Any]]:
    pcfg = cfg.get("pipeline", {})
    out = vis_tf.copy()
    info: Dict[str, Any] = {}
    nt = len(ctx.times_jd)
    dt = float(np.median(np.diff(ctx.times_jd)) * 86400.0) if nt > 1 else 1.0
    fr_hz = np.fft.fftshift(np.fft.fftfreq(nt, d=dt))
    fr_tf, _ = fringe_rate_transform(out, ctx.times_jd, ctx.weights_tf)

    notch = pcfg.get("fr_zero_notch", {})
    if bool(notch.get("enabled", False)):
        width_mhz = float(notch.get("width_mhz", 0.03))
        keep = np.abs(fr_hz * 1e3) > width_mhz
        fr_tf = fr_tf * keep[:, None]
        info["fr_zero_notch"] = {"enabled": True, "width_mhz": width_mhz, "kept_fraction": float(np.mean(keep))}
    else:
        info["fr_zero_notch"] = {"enabled": False}

    main = pcfg.get("mainlobe_fr_filter", {})
    if bool(main.get("enabled", False)):
        mode = main.get("mode", "remove_mainlobe")
        width_mhz = float(main.get("width_mhz", 0.8))
        if mode == "remove_mainlobe":
            keep = np.abs(fr_hz * 1e3) > width_mhz
        elif mode == "keep_mainlobe":
            keep = np.abs(fr_hz * 1e3) <= width_mhz
        else:
            raise ValueError(f"Unsupported mainlobe_fr_filter mode: {mode}")
        fr_tf = fr_tf * keep[:, None]
        info["mainlobe_fr_filter"] = {"enabled": True, "mode": mode, "width_mhz": width_mhz, "kept_fraction": float(np.mean(keep))}
    else:
        info["mainlobe_fr_filter"] = {"enabled": False}

    out = np.fft.ifft(np.fft.ifftshift(fr_tf, axes=0), axis=0)
    return out, info


def pipeline(vis_tf: np.ndarray, ctx: BackgroundContext, cfg: Dict[str, Any]) -> Tuple[np.ndarray, Dict[str, Any]]:
    # No redcal. No median flag replacement. Only weighted transforms.
    dly_out, dly_info = apply_delay_highpass_weighted(vis_tf, ctx, cfg)
    fr_out, fr_info = apply_fringe_filters(dly_out, ctx, cfg)
    return fr_out, {"delay_filter": dly_info, "fringe_filters": fr_info, "not_redcal": True, "no_flag_value_replacement": True}


def weighted_power(x: np.ndarray, w: Optional[np.ndarray] = None) -> float:
    if w is None:
        return float(np.nanmean(np.abs(x) ** 2))
    denom = max(float(np.nansum(w)), 1e-30)
    return float(np.nansum((np.abs(x) ** 2) * w) / denom)


def delay_excess_metric(vis_tf: np.ndarray, ctx: BackgroundContext, cfg: Dict[str, Any]) -> float:
    dc = cfg.get("metrics", {}).get("delay_excess", {})
    taper = make_taper(len(ctx.freqs_hz), dc.get("taper", cfg.get("pipeline", {}).get("delay_filter", {}).get("taper", "blackman_harris")), dc)
    dly = weighted_delay_transform(vis_tf, ctx.weights_tf, taper)
    delays = delay_axis_s(ctx.freqs_hz)
    horizon_ns = np.linalg.norm(ctx.baseline_enu_m) / C_M_PER_S * 1e9
    buffer_ns = float(dc.get("buffer_ns", 100.0))
    mask = np.abs(delays) >= (horizon_ns + buffer_ns) * 1e-9
    return float(np.nanmean(np.abs(dly[:, mask]) ** 2))


def local_delay_budget_metrics(proc_bg: np.ndarray, interaction: np.ndarray, ctx: BackgroundContext, cfg: Dict[str, Any]) -> Dict[str, Any]:
    dc = cfg.get("metrics", {}).get("delay_excess", {})
    local_cfg = cfg.get("metrics", {}).get("local_delay_budget", {})
    taper = make_taper(len(ctx.freqs_hz), dc.get("taper", cfg.get("pipeline", {}).get("delay_filter", {}).get("taper", "blackman_harris")), dc)
    delays = delay_axis_s(ctx.freqs_hz)
    delays_ns = delays * 1e9
    horizon_ns = np.linalg.norm(ctx.baseline_enu_m) / C_M_PER_S * 1e9
    buffer_ns = float(dc.get("buffer_ns", 100.0))
    eor_mask = np.abs(delays) >= (horizon_ns + buffer_ns) * 1e-9
    if not np.any(eor_mask):
        return {}

    bg_dly = weighted_delay_transform(proc_bg, ctx.weights_tf, taper)
    int_dly = weighted_delay_transform(interaction, ctx.weights_tf, taper)
    bg_power_tf = np.abs(bg_dly[:, eor_mask]) ** 2
    int_power_tf = np.abs(int_dly[:, eor_mask]) ** 2
    bg_profile = np.nanmean(bg_power_tf, axis=0)
    int_profile = np.nanmean(int_power_tf, axis=0)
    eps = 1e-30
    ratio_profile = int_profile / np.maximum(bg_profile, eps)
    top_percentile = float(local_cfg.get("top_percentile", 95.0))
    top_threshold = float(np.nanpercentile(int_profile, top_percentile))
    top_mask = int_profile >= top_threshold
    top_1_threshold = float(np.nanpercentile(int_profile, 99.0))
    top_1_mask = int_profile >= top_1_threshold
    top_5_mask = int_profile >= float(np.nanpercentile(int_profile, 95.0))
    peak_idx_local = int(np.nanargmax(ratio_profile))
    eor_delays_ns = delays_ns[eor_mask]

    per_time_int = np.nanmean(int_power_tf, axis=1)
    even = per_time_int[::2]
    odd = per_time_int[1::2]
    even_mean = float(np.nanmean(even)) if len(even) else np.nan
    odd_mean = float(np.nanmean(odd)) if len(odd) else np.nan
    pooled_std = float(np.nanstd(per_time_int))
    split_z = float(abs(even_mean - odd_mean) / max(pooled_std, eps)) if np.isfinite(pooled_std) else np.nan

    return {
        "local_eor_window_residual_vs_processed_background_db": float(10.0 * math.log10(max(float(np.nanmean(int_profile)), eps) / max(float(np.nanmean(bg_profile)), eps))),
        "local_peak_delay_bin_excess_db": float(10.0 * math.log10(max(float(np.nanmax(ratio_profile)), eps))),
        "local_peak_delay_ns": float(eor_delays_ns[peak_idx_local]),
        "local_top_percentile": top_percentile,
        "local_top_percentile_delay_bin_excess_db": float(10.0 * math.log10(max(float(np.nanmean(int_profile[top_mask])), eps) / max(float(np.nanmean(bg_profile[top_mask])), eps))),
        "top_1_percent_delay_bin_excess_db": float(10.0 * math.log10(max(float(np.nanmean(int_profile[top_1_mask])), eps) / max(float(np.nanmean(bg_profile[top_1_mask])), eps))),
        "top_5_percent_delay_bin_excess_db": float(10.0 * math.log10(max(float(np.nanmean(int_profile[top_5_mask])), eps) / max(float(np.nanmean(bg_profile[top_5_mask])), eps))),
        "local_null_split_even_residual_power": even_mean,
        "local_null_split_odd_residual_power": odd_mean,
        "local_null_split_residual_significance": split_z,
    }


def subspace_basis(x: np.ndarray, rank: int) -> np.ndarray:
    mat = np.nan_to_num(x.real) + 1j * np.nan_to_num(x.imag)
    # reshape already time x freq. SVD over time/frequency matrix.
    u, s, vh = np.linalg.svd(mat, full_matrices=False)
    k = min(rank, u.shape[1])
    return u[:, :k]


def subspace_overlap(a: np.ndarray, b: np.ndarray, rank: int = 5) -> float:
    ua = subspace_basis(a, rank)
    ub = subspace_basis(b, rank)
    return float(np.linalg.norm(ua.conj().T @ ub, ord="fro") / math.sqrt(min(ua.shape[1], ub.shape[1])))


def compute_metrics(raw_bg, raw_sat, raw_dirty, proc_bg, proc_sat, proc_dirty, interaction, ctx, cfg) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    rows = []
    products = {
        "raw_background": raw_bg,
        "raw_starlink": raw_sat,
        "raw_dirty": raw_dirty,
        "processed_background": proc_bg,
        "processed_starlink_only": proc_sat,
        "processed_dirty": proc_dirty,
        "interaction_residual": interaction,
    }
    bg_power = max(weighted_power(proc_bg, ctx.weights_tf), 1e-30)
    sat_raw_power = max(weighted_power(raw_sat, ctx.weights_tf), 1e-30)
    for name, arr in products.items():
        p = weighted_power(arr, ctx.weights_tf)
        rows.append({
            "product": name,
            "weighted_power": p,
            "power_vs_processed_background_db": 10 * math.log10(max(p, 1e-30) / bg_power),
            "power_vs_raw_starlink_db": 10 * math.log10(max(p, 1e-30) / sat_raw_power),
            "peak_abs": float(np.nanmax(np.abs(arr))),
            "mean_abs": float(np.nanmean(np.abs(arr))),
            "delay_excess_power": delay_excess_metric(arr, ctx, cfg),
        })
    metrics_df = pd.DataFrame(rows)

    rank = int(cfg.get("metrics", {}).get("subspace_rank", 5))
    sub_rows = [
        {"a": "processed_background", "b": "interaction_residual", "rank": rank, "overlap": subspace_overlap(proc_bg, interaction, rank)},
        {"a": "processed_starlink_only", "b": "interaction_residual", "rank": rank, "overlap": subspace_overlap(proc_sat, interaction, rank)},
        {"a": "processed_background", "b": "processed_starlink_only", "rank": rank, "overlap": subspace_overlap(proc_bg, proc_sat, rank)},
    ]
    if ctx.eor_tf is not None:
        eor_proc, _ = pipeline(ctx.eor_tf, ctx, cfg)
        sub_rows.append({"a": "processed_eor", "b": "interaction_residual", "rank": rank, "overlap": subspace_overlap(eor_proc, interaction, rank)})
    sub_df = pd.DataFrame(sub_rows)

    summary = {
        "interaction_vs_processed_background_db": float(metrics_df.loc[metrics_df["product"] == "interaction_residual", "power_vs_processed_background_db"].iloc[0]),
        "processed_starlink_survival_vs_raw_starlink_db": float(metrics_df.loc[metrics_df["product"] == "processed_starlink_only", "power_vs_raw_starlink_db"].iloc[0]),
        "interaction_survival_vs_raw_starlink_db": float(metrics_df.loc[metrics_df["product"] == "interaction_residual", "power_vs_raw_starlink_db"].iloc[0]),
        "subspace_rank": rank,
        **local_delay_budget_metrics(proc_bg, interaction, ctx, cfg),
    }
    return metrics_df, sub_df, summary




def slice_context(ctx: BackgroundContext, time_idx: np.ndarray) -> BackgroundContext:
    """Return a time-sliced background context for one analysis window."""
    time_idx = np.asarray(time_idx, dtype=int)
    eor = ctx.eor_tf[time_idx, :] if ctx.eor_tf is not None else None
    return BackgroundContext(
        vis_tf=ctx.vis_tf[time_idx, :],
        freqs_hz=ctx.freqs_hz,
        times_jd=ctx.times_jd[time_idx],
        baseline_enu_m=ctx.baseline_enu_m,
        weights_tf=ctx.weights_tf[time_idx, :],
        ant1_enu_m=ctx.ant1_enu_m,
        ant2_enu_m=ctx.ant2_enu_m,
        eor_tf=eor,
        source_format=ctx.source_format,
        source_path=ctx.source_path,
        metadata={**ctx.metadata, "time_slice_start": int(time_idx[0]), "time_slice_stop": int(time_idx[-1]) + 1},
    )


def enumerate_window_satellites(
    tle_path: Path,
    times_jd: np.ndarray,
    site_cfg: Dict[str, Any],
    window_hours: float = 1.0,
    min_alt_deg: float = 10.0,
    max_scan_satellites: int = 5000,
    min_samples_above_alt: int = 2,
    max_sats_per_window: Optional[int] = None,
    max_windows: Optional[int] = None,
    start_window: int = 0,
    peak_alt_min_deg: Optional[float] = None,
    peak_alt_max_deg: Optional[float] = None,
    cap_selection_method: str = "predicted_peak_apparent_flux",
    reference_flux_jy: float = 100.0,
    reference_range_km: float = 550.0,
    range_attenuation_mode: str = "flux_density_1_over_r2",
) -> Dict[int, Dict[str, Any]]:
    """Enumerate all catalog satellites visible in each fixed-duration window.

    Returns
    -------
    dict
        {window_idx: {"time_indices": ndarray, "satellites": [(sat, metadata), ...], ...}}

    Notes
    -----
    This scans the supplied TLE catalog over the exact time grid of the external
    background product. It is intentionally catalog-agnostic: pass a Starlink-only
    TLE file if the analysis should be Starlink-only.
    """
    if skyfield_load is None:
        raise ImportError(f"skyfield unavailable: {SKYFIELD_ERROR}")
    sats = skyfield_load.tle_file(str(tle_path))[:int(max_scan_satellites)]
    if not sats:
        raise ValueError(f"No satellites found in TLE file: {tle_path}")

    ts = skyfield_load.timescale()
    observer = skyfield_wgs84.latlon(
        latitude_degrees=float(site_cfg["lat_deg"]),
        longitude_degrees=float(site_cfg["lon_deg"]),
        elevation_m=float(site_cfg.get("elev_m", 0.0)),
    )
    times_jd = np.asarray(times_jd, dtype=float)
    rel_hours = (times_jd - times_jd[0]) * 24.0
    window_idx = np.floor(rel_hours / float(window_hours)).astype(int)
    windows: Dict[int, Dict[str, Any]] = {}

    processed_windows = 0
    for wid in sorted(np.unique(window_idx)):
        if int(wid) < int(start_window):
            continue
        if max_windows is not None and processed_windows >= int(max_windows):
            break
        tidx = np.flatnonzero(window_idx == wid)
        if len(tidx) == 0:
            continue
        processed_windows += 1
        t = ts.tt_jd(times_jd[tidx])
        entries = []
        for sidx, sat in enumerate(sats):
            try:
                topocentric = (sat - observer).at(t)
                alt_ang, _, distance = topocentric.altaz()
                alt = np.asarray(alt_ang.degrees, dtype=float)
                range_km = np.asarray(distance.km, dtype=float)
            except Exception:
                continue
            above = alt >= float(min_alt_deg)
            if int(np.sum(above)) < int(min_samples_above_alt):
                continue
            imax = int(np.nanargmax(alt))
            peak_alt_deg = float(alt[imax])
            range_at_peak_km = float(range_km[imax])
            range_min_km = float(np.nanmin(range_km))
            if range_attenuation_mode == "flux_density_1_over_r2":
                range_score = (float(reference_range_km) / max(range_at_peak_km, 1e-6)) ** 2
            elif range_attenuation_mode == "field_amplitude_1_over_r":
                range_score = float(reference_range_km) / max(range_at_peak_km, 1e-6)
            else:
                range_score = 1.0
            elevation_score = max(math.sin(math.radians(peak_alt_deg)), 0.0) ** 2
            predicted_peak_apparent_flux_jy = float(reference_flux_jy) * float(range_score) * float(elevation_score)
            entries.append((
                sat,
                {
                    "satellite_name": getattr(sat, "name", f"sat_{sidx}"),
                    "satellite_index": int(sidx),
                    "window_idx": int(wid),
                    "peak_alt_deg": peak_alt_deg,
                    "peak_jd": float(times_jd[tidx[imax]]),
                    "range_at_peak_km": range_at_peak_km,
                    "range_min_km_predicted": range_min_km,
                    "predicted_peak_apparent_flux_jy": predicted_peak_apparent_flux_jy,
                    "samples_above_min_alt": int(np.sum(above)),
                    "min_alt_deg": float(min_alt_deg),
                },
            ))
        n_visible_total = int(len(entries))
        if peak_alt_min_deg is not None:
            entries = [entry for entry in entries if entry[1]["peak_alt_deg"] >= float(peak_alt_min_deg)]
        if peak_alt_max_deg is not None:
            entries = [entry for entry in entries if entry[1]["peak_alt_deg"] <= float(peak_alt_max_deg)]
        n_after_peak_alt_gate = int(len(entries))
        if cap_selection_method == "predicted_peak_apparent_flux":
            score_key = "predicted_peak_apparent_flux_jy"
        elif cap_selection_method == "peak_alt_deg":
            score_key = "peak_alt_deg"
        else:
            raise ValueError(
                "Unsupported multi_window.cap_selection_method: "
                f"{cap_selection_method}. Use predicted_peak_apparent_flux or peak_alt_deg."
            )
        for _, meta in entries:
            meta["cap_selection_method"] = cap_selection_method
            meta["cap_selection_score"] = float(meta[score_key])
        entries.sort(key=lambda x: x[1]["cap_selection_score"], reverse=True)
        truncated_by_cap = bool(max_sats_per_window is not None and n_after_peak_alt_gate > int(max_sats_per_window))
        if max_sats_per_window is not None:
            entries = entries[: int(max_sats_per_window)]
        windows[int(wid)] = {
            "time_indices": tidx,
            "start_jd": float(times_jd[tidx[0]]),
            "stop_jd": float(times_jd[tidx[-1]]),
            "duration_hours": float((times_jd[tidx[-1]] - times_jd[tidx[0]]) * 24.0),
            "satellites": entries,
            "n_visible_total": n_visible_total,
            "n_after_peak_alt_gate": n_after_peak_alt_gate,
            "n_used_emitters": int(len(entries)),
            "max_sats_per_window": None if max_sats_per_window is None else int(max_sats_per_window),
            "truncated_by_cap": truncated_by_cap,
            "peak_alt_min_deg": None if peak_alt_min_deg is None else float(peak_alt_min_deg),
            "peak_alt_max_deg": None if peak_alt_max_deg is None else float(peak_alt_max_deg),
            "cap_selection_method": cap_selection_method,
        }
    return windows


def multi_satellite_injection(sat_entries, ctx: BackgroundContext, cfg: Dict[str, Any], out_dir: Path, window_idx: int = 0):
    """Build complex-summed visibility for all satellites in a window.

    The sum is complex and phase-preserving:
        V_total(t, nu) = Σ_k V_k(t, nu)
    """
    per_vis = []
    per_tracks = []
    per_reports = []
    total = np.zeros_like(ctx.vis_tf, dtype=complex)
    for k, (sat, meta) in enumerate(sat_entries):
        vis, track, report, arrays = build_starlink_visibility(
            ctx,
            cfg,
            out_dir,
            sat_override=sat,
            selected_override=meta,
            random_seed_offset=1000 * int(window_idx) + k,
        )
        total = total + vis
        per_vis.append(vis)
        tr = track.copy()
        tr["satellite_name"] = meta.get("satellite_name", f"sat_{k}")
        tr["satellite_index"] = meta.get("satellite_index", k)
        per_tracks.append(tr)
        per_reports.append({**report, "window_idx": int(window_idx), "local_satellite_number": int(k)})
    return total, per_vis, per_tracks, per_reports


def pairwise_cross_correlation(per_sat_vis_list, ctx: BackgroundContext, cfg: Dict[str, Any], per_sat_reports=None, per_tracks=None) -> pd.DataFrame:
    """Compute pairwise delay-domain coherence coefficients.

    Γ_ij(τ) = <D_i*(t,τ) D_j(t,τ)>_t
    C_ij(τ) = |Γ_ij(τ)| / sqrt(<|D_i|²>_t <|D_j|²>_t)
    """
    if len(per_sat_vis_list) < 2:
        return pd.DataFrame(columns=[
            "i", "j", "satellite_i", "satellite_j", "peak_coherence", "weighted_coherence",
            "peak_delay_ns", "min_delta_tau_ns", "median_delta_tau_ns",
            "delay_bin_width_ns", "fraction_time_delta_tau_lt_bin",
            "fraction_time_delta_tau_lt_half_bin", "overlap_duration_same_delay_bin_s",
            "p10_delta_tau_ns", "combined_predicted_apparent_flux_jy",
            "mean_pair_alt_deg", "mean_pair_inverse_range_km",
        ])
    dc = cfg.get("metrics", {}).get("delay_excess", {})
    pair_cfg = cfg.get("metrics", {}).get("pairwise_delay_overlap", {})
    taper = make_taper(len(ctx.freqs_hz), dc.get("taper", cfg.get("pipeline", {}).get("delay_filter", {}).get("taper", "blackman_harris")), dc)
    delays_ns = delay_axis_s(ctx.freqs_hz) * 1e9
    configured_bin_ns = pair_cfg.get("delay_bin_width_ns")
    delay_bin_width_ns = (
        abs(float(np.nanmedian(np.diff(delays_ns)))) if configured_bin_ns is None and len(delays_ns) > 1
        else float(configured_bin_ns if configured_bin_ns is not None else 0.0)
    )
    dt_s = float(np.nanmedian(np.diff(ctx.times_jd)) * 86400.0) if len(ctx.times_jd) > 1 else 0.0
    dlys = [weighted_delay_transform(v, ctx.weights_tf, taper) for v in per_sat_vis_list]
    names = []
    for k in range(len(per_sat_vis_list)):
        if per_sat_reports and k < len(per_sat_reports):
            names.append(per_sat_reports[k].get("selected_satellite", {}).get("satellite_name", f"sat_{k}"))
        else:
            names.append(f"sat_{k}")
    rows = []
    eps = 1e-30
    for i in range(len(dlys)):
        pi_tau = np.nanmean(np.abs(dlys[i]) ** 2, axis=0)
        for j in range(i + 1, len(dlys)):
            pj_tau = np.nanmean(np.abs(dlys[j]) ** 2, axis=0)
            gamma = np.nanmean(np.conj(dlys[i]) * dlys[j], axis=0)
            pair_weight = np.sqrt(np.maximum(pi_tau * pj_tau, eps))
            coeff = np.abs(gamma) / pair_weight
            peak_idx = int(np.nanargmax(coeff))
            weighted_coherence = float(np.nansum(coeff * pair_weight) / max(float(np.nansum(pair_weight)), eps))
            min_delta_tau_ns = np.nan
            median_delta_tau_ns = np.nan
            p10_delta_tau_ns = np.nan
            fraction_time_delta_tau_lt_bin = np.nan
            fraction_time_delta_tau_lt_half_bin = np.nan
            overlap_duration_same_delay_bin_s = np.nan
            mean_pair_alt_deg = np.nan
            mean_pair_inverse_range_km = np.nan
            if per_tracks is not None and i < len(per_tracks) and j < len(per_tracks):
                if "tau_s" in per_tracks[i].columns and "tau_s" in per_tracks[j].columns:
                    delta_tau_ns = np.abs(
                        per_tracks[i]["tau_s"].to_numpy(dtype=float)
                        - per_tracks[j]["tau_s"].to_numpy(dtype=float)
                    ) * 1e9
                    min_delta_tau_ns = float(np.nanmin(delta_tau_ns))
                    median_delta_tau_ns = float(np.nanmedian(delta_tau_ns))
                    p10_delta_tau_ns = float(np.nanpercentile(delta_tau_ns, 10))
                    same_bin = delta_tau_ns < delay_bin_width_ns
                    same_half_bin = delta_tau_ns < 0.5 * delay_bin_width_ns
                    fraction_time_delta_tau_lt_bin = float(np.nanmean(same_bin))
                    fraction_time_delta_tau_lt_half_bin = float(np.nanmean(same_half_bin))
                    overlap_duration_same_delay_bin_s = float(np.nansum(same_bin) * dt_s)
                if "alt_deg" in per_tracks[i].columns and "alt_deg" in per_tracks[j].columns:
                    mean_pair_alt_deg = float(0.5 * (
                        np.nanmean(per_tracks[i]["alt_deg"].to_numpy(dtype=float))
                        + np.nanmean(per_tracks[j]["alt_deg"].to_numpy(dtype=float))
                    ))
                if "range_km" in per_tracks[i].columns and "range_km" in per_tracks[j].columns:
                    mean_pair_inverse_range_km = float(0.5 * (
                        np.nanmean(1.0 / np.clip(per_tracks[i]["range_km"].to_numpy(dtype=float), 1e-6, None))
                        + np.nanmean(1.0 / np.clip(per_tracks[j]["range_km"].to_numpy(dtype=float), 1e-6, None))
                    ))
            combined_predicted_flux = np.nan
            if per_sat_reports and i < len(per_sat_reports) and j < len(per_sat_reports):
                si = per_sat_reports[i].get("selected_satellite", {})
                sj = per_sat_reports[j].get("selected_satellite", {})
                fi = si.get("predicted_peak_apparent_flux_jy", np.nan)
                fj = sj.get("predicted_peak_apparent_flux_jy", np.nan)
                combined_predicted_flux = float(fi) + float(fj)
            rows.append({
                "i": int(i),
                "j": int(j),
                "satellite_i": names[i],
                "satellite_j": names[j],
                "peak_coherence": float(coeff[peak_idx]),
                "weighted_coherence": weighted_coherence,
                "peak_delay_ns": float(delays_ns[peak_idx]),
                "min_delta_tau_ns": min_delta_tau_ns,
                "median_delta_tau_ns": median_delta_tau_ns,
                "p10_delta_tau_ns": p10_delta_tau_ns,
                "delay_bin_width_ns": delay_bin_width_ns,
                "fraction_time_delta_tau_lt_bin": fraction_time_delta_tau_lt_bin,
                "fraction_time_delta_tau_lt_half_bin": fraction_time_delta_tau_lt_half_bin,
                "overlap_duration_same_delay_bin_s": overlap_duration_same_delay_bin_s,
                "combined_predicted_apparent_flux_jy": combined_predicted_flux,
                "mean_pair_alt_deg": mean_pair_alt_deg,
                "mean_pair_inverse_range_km": mean_pair_inverse_range_km,
                "mean_coherence": float(np.nanmean(coeff)),
                "p95_coherence": float(np.nanpercentile(coeff, 95)),
            })
    return pd.DataFrame(rows)


def phase_randomized_null_ratios(per_sat_vis_list, ctx: BackgroundContext, cfg: Dict[str, Any], n_trials: int, seed: int) -> np.ndarray:
    if len(per_sat_vis_list) == 0 or n_trials <= 0:
        return np.asarray([], dtype=float)
    p_each = np.array([delay_excess_metric(v, ctx, cfg) for v in per_sat_vis_list], dtype=float)
    p_incoh = float(np.nansum(p_each))
    rng = np.random.default_rng(int(seed))
    ratios = np.empty(int(n_trials), dtype=float)
    stack = np.stack(per_sat_vis_list, axis=0)
    for t in range(int(n_trials)):
        phases = np.exp(1j * rng.uniform(0.0, 2.0 * np.pi, size=len(per_sat_vis_list)))
        null_total = np.sum(stack * phases[:, None, None], axis=0)
        ratios[t] = delay_excess_metric(null_total, ctx, cfg) / max(p_incoh, 1e-30)
    return ratios


def coherence_ratio_per_window(total_vis: np.ndarray, per_sat_vis_list, ctx: BackgroundContext, cfg: Dict[str, Any], window_idx: int = 0) -> Tuple[Dict[str, Any], np.ndarray]:
    """Compare complex-summed delay-excess power to incoherent expectation.

    ratio > 1 means the complex sum produces more delay-excess power than the
    sum of independently processed single-satellite powers.
    """
    p_coh = delay_excess_metric(total_vis, ctx, cfg)
    p_each = np.array([delay_excess_metric(v, ctx, cfg) for v in per_sat_vis_list], dtype=float)
    p_incoh = float(np.nansum(p_each))
    ratio = float(p_coh / max(p_incoh, 1e-30))
    null_cfg = cfg.get("metrics", {}).get("phase_randomized_null", {})
    n_trials = int(null_cfg.get("n_trials", cfg.get("multi_window", {}).get("phase_randomized_null_trials", 256)))
    base_seed = int(cfg.get("experiment", {}).get("random_seed", 0)) + 7919 * int(window_idx)
    null_ratios = phase_randomized_null_ratios(per_sat_vis_list, ctx, cfg, n_trials=n_trials, seed=base_seed)
    if len(null_ratios):
        null_median = float(np.nanmedian(null_ratios))
        null_p95 = float(np.nanpercentile(null_ratios, 95))
        null_p99 = float(np.nanpercentile(null_ratios, 99))
        null_mean = float(np.nanmean(null_ratios))
        observed_exceeds_null_p95 = bool(ratio > null_p95)
        observed_minus_null_median_db = float(10.0 * math.log10(max(ratio, 1e-30) / max(null_median, 1e-30)))
    else:
        null_median = null_p95 = null_p99 = null_mean = observed_minus_null_median_db = np.nan
        observed_exceeds_null_p95 = False
    return {
        "delay_excess_power_coherent_complex_sum": float(p_coh),
        "delay_excess_power_incoherent_sum": float(p_incoh),
        "coherence_amplification_ratio": ratio,
        "coherence_amplification_db": float(10.0 * math.log10(max(ratio, 1e-30))),
        "phase_randomized_null_trials": int(len(null_ratios)),
        "phase_randomized_null_ratio_mean": null_mean,
        "phase_randomized_null_ratio_median": null_median,
        "phase_randomized_null_ratio_p95": null_p95,
        "phase_randomized_null_ratio_p99": null_p99,
        "observed_exceeds_phase_null_p95": observed_exceeds_null_p95,
        "observed_minus_phase_null_median_db": observed_minus_null_median_db,
        "n_emitters": int(len(per_sat_vis_list)),
        "max_single_delay_excess_power": float(np.nanmax(p_each)) if len(p_each) else 0.0,
        "median_single_delay_excess_power": float(np.nanmedian(p_each)) if len(p_each) else 0.0,
    }, null_ratios


def run_multi_window_analysis(ctx: BackgroundContext, cfg: Dict[str, Any], out_dir: Path) -> None:
    mw = cfg.get("multi_window", {})
    sat_cfg = cfg.get("starlink", {})
    tle_path = Path(sat_cfg["tle_path"])
    max_windows = mw.get("max_windows")
    start_window = int(mw.get("start_window", 0))
    windows = enumerate_window_satellites(
        tle_path=tle_path,
        times_jd=ctx.times_jd,
        site_cfg=cfg["site"],
        window_hours=float(mw.get("window_hours", 1.0)),
        min_alt_deg=float(mw.get("min_alt_deg", 10.0)),
        max_scan_satellites=int(mw.get("max_scan_satellites", sat_cfg.get("max_scan_satellites", 5000))),
        min_samples_above_alt=int(mw.get("min_samples_above_alt", 2)),
        max_sats_per_window=mw.get("max_sats_per_window"),
        max_windows=max_windows,
        start_window=start_window,
        peak_alt_min_deg=mw.get("peak_alt_min_deg"),
        peak_alt_max_deg=mw.get("peak_alt_max_deg"),
        cap_selection_method=str(mw.get("cap_selection_method", "predicted_peak_apparent_flux")),
        reference_flux_jy=float(sat_cfg.get("reference_flux_jy", sat_cfg.get("flux_jy_ref", 100.0))),
        reference_range_km=float(sat_cfg.get("reference_range_km", 550.0)),
        range_attenuation_mode=str(sat_cfg.get("range_attenuation_mode", "flux_density_1_over_r2")),
    )

    all_window_rows = []
    all_pair_rows = []
    all_sat_reports = []
    all_tracks = []
    all_null_rows = []

    def flush_partial_outputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        window_df_local = pd.DataFrame(all_window_rows)
        pair_all_local = pd.concat(all_pair_rows, ignore_index=True) if all_pair_rows else pd.DataFrame()
        sat_df_local = pd.DataFrame(all_sat_reports)
        track_df_local = pd.concat(all_tracks, ignore_index=True) if all_tracks else pd.DataFrame()
        null_df_local = pd.DataFrame(all_null_rows)
        window_df_local.to_csv(out_dir / "tables" / "multi_window_summary.csv", index=False)
        pair_all_local.to_csv(out_dir / "tables" / "pairwise_delay_coherence.csv", index=False)
        sat_df_local.to_csv(out_dir / "tables" / "per_satellite_window_summary.csv", index=False)
        track_df_local.to_csv(out_dir / "tables" / "multi_window_tracks.csv", index=False)
        null_df_local.to_csv(out_dir / "tables" / "phase_randomized_null_ratios.csv", index=False)
        return window_df_local, pair_all_local, sat_df_local, track_df_local

    for wid, win in windows.items():
        sat_entries = win["satellites"]
        if not sat_entries:
            all_window_rows.append({
                "window_idx": int(wid),
                "n_visible_total": int(win.get("n_visible_total", 0)),
                "n_after_peak_alt_gate": int(win.get("n_after_peak_alt_gate", 0)),
                "n_used_emitters": 0,
                "max_sats_per_window": win.get("max_sats_per_window"),
                "truncated_by_cap": bool(win.get("truncated_by_cap", False)),
                "peak_alt_min_deg": win.get("peak_alt_min_deg"),
                "peak_alt_max_deg": win.get("peak_alt_max_deg"),
                "cap_selection_method": win.get("cap_selection_method"),
                "selected_pol": selected_polarization_product(ctx, cfg),
                "polarization_mode": cfg.get("starlink", {}).get("polarization", {}).get("mode"),
                "polarization_contrast": cfg.get("starlink", {}).get("polarization", {}).get("contrast"),
                "n_emitters": 0,
                "status": "no_visible_satellites",
            })
            flush_partial_outputs()
            continue
        wctx = slice_context(ctx, win["time_indices"])
        total_sat, per_vis, per_tracks, per_reports = multi_satellite_injection(sat_entries, wctx, cfg, out_dir, window_idx=int(wid))
        raw_bg = wctx.vis_tf
        raw_dirty = raw_bg + total_sat
        proc_bg, _ = pipeline(raw_bg, wctx, cfg)
        proc_sat, _ = pipeline(total_sat, wctx, cfg)
        proc_dirty, _ = pipeline(raw_dirty, wctx, cfg)
        interaction = proc_dirty - proc_bg
        metrics_df, sub_df, summary = compute_metrics(raw_bg, total_sat, raw_dirty, proc_bg, proc_sat, proc_dirty, interaction, wctx, cfg)
        coh_stats, null_ratios = coherence_ratio_per_window(total_sat, per_vis, wctx, cfg, window_idx=int(wid))
        for trial_idx, ratio_value in enumerate(null_ratios):
            all_null_rows.append({
                "window_idx": int(wid),
                "trial": int(trial_idx),
                "phase_randomized_coherence_amplification_ratio": float(ratio_value),
                "phase_randomized_coherence_amplification_db": float(10.0 * math.log10(max(float(ratio_value), 1e-30))),
                "observed_coherence_amplification_ratio": float(coh_stats["coherence_amplification_ratio"]),
                "observed_coherence_amplification_db": float(coh_stats["coherence_amplification_db"]),
            })
        pair_df = pairwise_cross_correlation(per_vis, wctx, cfg, per_reports, per_tracks)
        if not pair_df.empty:
            pair_df.insert(0, "window_idx", int(wid))
            all_pair_rows.append(pair_df)
        for rep in per_reports:
            flat = {
                "window_idx": int(wid),
                "satellite_name": rep.get("selected_satellite", {}).get("satellite_name"),
                "satellite_index": rep.get("selected_satellite", {}).get("satellite_index"),
                "peak_alt_deg": rep.get("peak_alt_deg"),
                "range_min_km": rep.get("range_min_km"),
                "range_max_km": rep.get("range_max_km"),
                "peak_abs_jy": rep.get("peak_abs_jy"),
                "mean_abs_jy": rep.get("mean_abs_jy"),
                "predicted_peak_apparent_flux_jy": rep.get("selected_satellite", {}).get("predicted_peak_apparent_flux_jy"),
                "cap_selection_method": rep.get("selected_satellite", {}).get("cap_selection_method"),
                "cap_selection_score": rep.get("selected_satellite", {}).get("cap_selection_score"),
                "emission_type": rep.get("emission_type"),
            }
            all_sat_reports.append(flat)
        for tr in per_tracks:
            tr2 = tr.copy()
            tr2.insert(0, "window_idx", int(wid))
            all_tracks.append(tr2)
        first_pol_meta = per_reports[0].get("polarization", {}) if per_reports else {}
        first_pol_diag = first_pol_meta.get("diagnostics", {}) if isinstance(first_pol_meta, dict) else {}
        row = {
            "window_idx": int(wid),
            "status": "ok",
            "start_jd": win["start_jd"],
            "stop_jd": win["stop_jd"],
            "duration_hours": win["duration_hours"],
            "n_visible_total": int(win.get("n_visible_total", len(per_vis))),
            "n_after_peak_alt_gate": int(win.get("n_after_peak_alt_gate", len(per_vis))),
            "n_used_emitters": int(len(per_vis)),
            "max_sats_per_window": win.get("max_sats_per_window"),
            "truncated_by_cap": bool(win.get("truncated_by_cap", False)),
            "peak_alt_min_deg": win.get("peak_alt_min_deg"),
            "peak_alt_max_deg": win.get("peak_alt_max_deg"),
            "cap_selection_method": win.get("cap_selection_method"),
            "selected_pol": first_pol_meta.get("selected_pol") if isinstance(first_pol_meta, dict) else None,
            "polarization_mode": first_pol_meta.get("mode") if isinstance(first_pol_meta, dict) else None,
            "polarization_contrast": first_pol_meta.get("contrast") if isinstance(first_pol_meta, dict) else None,
            "polarization_jones_model": first_pol_meta.get("jones_model") if isinstance(first_pol_meta, dict) else None,
            "polarization_stokes_Q_over_I_mean_abs": first_pol_diag.get("stokes_Q_over_I_mean_abs") if isinstance(first_pol_diag, dict) else None,
            "XX_YY_power_ratio": (
                float(first_pol_diag.get("power_XX")) / max(float(first_pol_diag.get("power_YY")), 1e-30)
                if isinstance(first_pol_diag, dict) and first_pol_diag.get("power_XX") is not None and first_pol_diag.get("power_YY") is not None
                else None
            ),
            "n_emitters": int(len(per_vis)),
            **summary,
            **coh_stats,
            "max_pair_peak_coherence": float(pair_df["peak_coherence"].max()) if not pair_df.empty else np.nan,
            "max_pair_weighted_coherence": float(pair_df["weighted_coherence"].max()) if not pair_df.empty else np.nan,
            "median_pair_peak_coherence": float(pair_df["peak_coherence"].median()) if not pair_df.empty else np.nan,
            "median_pair_weighted_coherence": float(pair_df["weighted_coherence"].median()) if not pair_df.empty else np.nan,
            "min_pair_delta_tau_ns": float(pair_df["min_delta_tau_ns"].min()) if not pair_df.empty else np.nan,
            "median_pair_delta_tau_ns": float(pair_df["median_delta_tau_ns"].median()) if not pair_df.empty else np.nan,
            "max_pair_fraction_time_delta_tau_lt_bin": float(pair_df["fraction_time_delta_tau_lt_bin"].max()) if not pair_df.empty else np.nan,
            "median_pair_fraction_time_delta_tau_lt_bin": float(pair_df["fraction_time_delta_tau_lt_bin"].median()) if not pair_df.empty else np.nan,
            "max_pair_overlap_duration_same_delay_bin_s": float(pair_df["overlap_duration_same_delay_bin_s"].max()) if not pair_df.empty else np.nan,
        }
        all_window_rows.append(row)
        if bool(mw.get("save_window_arrays", False)):
            wdir = out_dir / "arrays" / f"window_{int(wid):04d}"
            ensure_dir(wdir)
            np.save(wdir / "raw_background.npy", raw_bg)
            np.save(wdir / "raw_starlink_total.npy", total_sat)
            np.save(wdir / "raw_dirty.npy", raw_dirty)
            np.save(wdir / "processed_background.npy", proc_bg)
            np.save(wdir / "processed_starlink_only.npy", proc_sat)
            np.save(wdir / "processed_dirty.npy", proc_dirty)
            np.save(wdir / "interaction_residual.npy", interaction)
            np.save(wdir / "weights_tf.npy", wctx.weights_tf)
        flush_partial_outputs()

    window_df, pair_all, sat_df, track_df = flush_partial_outputs()

    # Simple paper-facing diagnostic plot.
    if not window_df.empty and "coherence_amplification_db" in window_df.columns:
        fig, ax = plt.subplots(figsize=(8, 4))
        ok = window_df[window_df.get("status", "ok") == "ok"].copy()
        if not ok.empty:
            x_col = "n_used_emitters" if "n_used_emitters" in ok.columns else "n_emitters"
            ax.scatter(ok[x_col], ok["coherence_amplification_db"])
            for _, r in ok.iterrows():
                label = f"w{int(r['window_idx'])}"
                if bool(r.get("truncated_by_cap", False)):
                    label += f" ({int(r.get('n_visible_total', r[x_col]))} visible)"
                ax.annotate(label, (r[x_col], r["coherence_amplification_db"]), fontsize=7, alpha=0.7)
        ax.axhline(0, color="k", linestyle=":", linewidth=1)
        ax.set_xlabel("Used emitters per window after cap")
        ax.set_ylabel("Coherence amplification [dB]\nP(|ΣV_k|) / ΣP(V_k)")
        ax.set_title("Multi-satellite coherent amplification by window")
        ax.grid(alpha=0.3)
        fig.savefig(out_dir / "figures" / "fig04_multi_window_coherence_amplification.png", dpi=180, bbox_inches="tight")
        plt.close(fig)

    if not pair_all.empty and {"fraction_time_delta_tau_lt_bin", "weighted_coherence"}.issubset(pair_all.columns):
        fig, ax = plt.subplots(figsize=(7.5, 4.5))
        color_col = "combined_predicted_apparent_flux_jy"
        size_col = "mean_pair_alt_deg"
        c = pair_all[color_col] if color_col in pair_all.columns else None
        if size_col in pair_all.columns:
            s = 18.0 + 1.8 * np.clip(pair_all[size_col].to_numpy(dtype=float), 0.0, 90.0)
        else:
            s = 36
        sc = ax.scatter(
            pair_all["fraction_time_delta_tau_lt_bin"],
            pair_all["weighted_coherence"],
            c=c,
            s=s,
            alpha=0.75,
            cmap="viridis",
            label="weighted coherence",
        )
        ax.scatter(pair_all["fraction_time_delta_tau_lt_bin"], pair_all["peak_coherence"], s=12, alpha=0.25, color="tab:gray", label="peak coherence")
        if c is not None:
            fig.colorbar(sc, ax=ax, label="Combined predicted apparent flux [Jy]")
        ax.set_xlabel("Fraction of time with |Delta tau| below one delay bin")
        ax.set_ylabel("Delay-domain coherence")
        ax.set_title("Pairwise same-delay-bin overlap vs coherence")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
        fig.savefig(out_dir / "figures" / "fig05_pairwise_delay_separation_vs_coherence.png", dpi=180, bbox_inches="tight")
        plt.close(fig)

    report = {
        "multi_window_enabled": True,
        "window_hours": float(mw.get("window_hours", 1.0)),
        "min_alt_deg": float(mw.get("min_alt_deg", 10.0)),
        "n_windows": int(len(windows)),
        "n_windows_with_emitters": int(np.sum(window_df.get("n_emitters", pd.Series(dtype=int)).fillna(0).to_numpy() > 0)) if not window_df.empty else 0,
        "outputs": [
            "tables/multi_window_summary.csv",
            "tables/pairwise_delay_coherence.csv",
            "tables/per_satellite_window_summary.csv",
            "tables/multi_window_tracks.csv",
            "figures/fig04_multi_window_coherence_amplification.png",
            "figures/fig05_pairwise_delay_separation_vs_coherence.png",
        ],
        "interpretation": (
            "ratio > 1 indicates complex-sum delay-excess power above incoherent sum over individual emitters. "
            "Compare observed amplification against phase-randomized null columns before making a coherent-overlap claim."
        ),
    }
    with open(out_dir / "reports" / "multi_window_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

def plot_waterfall(path: Path, arrays: Dict[str, np.ndarray], freqs_hz: np.ndarray):
    n = len(arrays)
    fig, axes = plt.subplots(1, n, figsize=(4.6 * n, 4.0), constrained_layout=True)
    if n == 1:
        axes = [axes]
    vmax = max(float(np.nanpercentile(np.abs(a), 99.5)) for a in arrays.values())
    vmax = max(vmax, 1e-30)
    extent = [freqs_hz[0] / 1e6, freqs_hz[-1] / 1e6, 0, next(iter(arrays.values())).shape[0]]
    im = None
    for ax, (name, arr) in zip(axes, arrays.items()):
        im = ax.imshow(np.abs(arr), aspect="auto", origin="lower", extent=extent, vmin=0, vmax=vmax)
        ax.set_title(name)
        ax.set_xlabel("Frequency [MHz]")
        ax.set_ylabel("Time index")
    fig.colorbar(im, ax=axes, shrink=0.9, label="|V|")
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_delay_profiles(path: Path, products: Dict[str, np.ndarray], ctx: BackgroundContext, cfg: Dict[str, Any]):
    dc = cfg.get("metrics", {}).get("delay_excess", {})
    taper = make_taper(len(ctx.freqs_hz), dc.get("taper", "blackman_harris"), dc)
    delays_ns = delay_axis_s(ctx.freqs_hz) * 1e9
    fig, ax = plt.subplots(figsize=(8.0, 4.5))
    for name, arr in products.items():
        dly = weighted_delay_transform(arr, ctx.weights_tf, taper)
        ax.plot(delays_ns, np.nanmean(np.abs(dly), axis=0), label=name)
    horizon_ns = np.linalg.norm(ctx.baseline_enu_m) / C_M_PER_S * 1e9
    buffer_ns = float(dc.get("buffer_ns", 100.0))
    for x in [-(horizon_ns + buffer_ns), horizon_ns + buffer_ns]:
        ax.axvline(x, color="k", linestyle=":", linewidth=1)
    ax.set_xlabel("Delay [ns]")
    ax.set_ylabel("Mean weighted |delay transform|")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    cfg = read_yaml(args.config)
    if args.background is not None:
        cfg.setdefault("background", {})["path"] = str(args.background)
    if args.tle_path is not None:
        cfg.setdefault("starlink", {})["tle_path"] = str(args.tle_path)
    if args.pol is not None:
        cfg.setdefault("uvh5_selection", {})["pol"] = str(args.pol)
    if args.start_window is not None:
        cfg.setdefault("multi_window", {})["start_window"] = int(args.start_window)
    if args.max_windows is not None:
        cfg.setdefault("multi_window", {})["max_windows"] = int(args.max_windows)
    if args.max_sats_per_window is not None:
        cfg.setdefault("multi_window", {})["max_sats_per_window"] = int(args.max_sats_per_window)
    if args.max_scan_satellites is not None:
        cfg.setdefault("multi_window", {})["max_scan_satellites"] = int(args.max_scan_satellites)
    if args.peak_alt_min_deg is not None:
        cfg.setdefault("multi_window", {})["peak_alt_min_deg"] = float(args.peak_alt_min_deg)
    if args.peak_alt_max_deg is not None:
        cfg.setdefault("multi_window", {})["peak_alt_max_deg"] = float(args.peak_alt_max_deg)
    if args.cap_selection_method is not None:
        cfg.setdefault("multi_window", {})["cap_selection_method"] = str(args.cap_selection_method)
    if args.polarization_mode is not None:
        cfg.setdefault("starlink", {}).setdefault("polarization", {})["mode"] = str(args.polarization_mode)
    if args.polarization_contrast is not None:
        cfg.setdefault("starlink", {}).setdefault("polarization", {})["contrast"] = float(args.polarization_contrast)
    if args.save_window_arrays:
        cfg.setdefault("multi_window", {})["save_window_arrays"] = True
    if args.phase_null_trials is not None:
        cfg.setdefault("metrics", {}).setdefault("phase_randomized_null", {})["n_trials"] = int(args.phase_null_trials)
        cfg.setdefault("multi_window", {})["phase_randomized_null_trials"] = int(args.phase_null_trials)
    out_dir = args.output_dir or Path(cfg.get("experiment", {}).get("output_dir", "nearfield_starlink_on_background_outputs"))
    out_dir = Path(out_dir)
    for sub in ["arrays", "tables", "figures", "reports"]:
        ensure_dir(out_dir / sub)

    bg_path = Path(cfg.get("background", {}).get("path", ""))
    if not bg_path.exists():
        raise FileNotFoundError(
            "External background visibility is required. Provide background.path in config or --background. "
            "This script intentionally does not create an internal toy foreground."
        )
    ctx = load_background(bg_path, cfg)
    if bool(cfg.get("multi_window", {}).get("enabled", False)):
        run_multi_window_analysis(ctx, cfg, out_dir)
        with open(out_dir / "config_used.yaml", "w", encoding="utf-8") as f:
            yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=True)
        print(json.dumps({"output_dir": str(out_dir), "multi_window_enabled": True}, ensure_ascii=False, indent=2))
        return

    raw_background = ctx.vis_tf
    raw_starlink, track, inj_report, inj_arrays = build_starlink_visibility(ctx, cfg, out_dir)
    raw_dirty = raw_background + raw_starlink

    processed_background, pipe_bg = pipeline(raw_background, ctx, cfg)
    processed_starlink_only, pipe_sat = pipeline(raw_starlink, ctx, cfg)
    processed_dirty, pipe_dirty = pipeline(raw_dirty, ctx, cfg)
    interaction_residual = processed_dirty - processed_background

    metrics_df, sub_df, summary = compute_metrics(
        raw_background, raw_starlink, raw_dirty,
        processed_background, processed_starlink_only, processed_dirty, interaction_residual,
        ctx, cfg,
    )

    # Save arrays.
    np.save(out_dir / "arrays" / "raw_background.npy", raw_background)
    np.save(out_dir / "arrays" / "raw_starlink.npy", raw_starlink)
    np.save(out_dir / "arrays" / "raw_dirty.npy", raw_dirty)
    np.save(out_dir / "arrays" / "processed_background.npy", processed_background)
    np.save(out_dir / "arrays" / "processed_starlink_only.npy", processed_starlink_only)
    np.save(out_dir / "arrays" / "processed_dirty.npy", processed_dirty)
    np.save(out_dir / "arrays" / "interaction_residual.npy", interaction_residual)
    np.save(out_dir / "arrays" / "weights_tf.npy", ctx.weights_tf)
    np.savez_compressed(out_dir / "arrays" / "starlink_components.npz", **inj_arrays)

    track.to_csv(out_dir / "tables" / "starlink_track.csv", index=False)
    metrics_df.to_csv(out_dir / "tables" / "summary_metrics.csv", index=False)
    sub_df.to_csv(out_dir / "tables" / "subspace_overlap.csv", index=False)

    # Plots.
    plot_waterfall(out_dir / "figures" / "fig01_raw_components.png", {
        "background": raw_background,
        "starlink": raw_starlink,
        "dirty": raw_dirty,
    }, ctx.freqs_hz)
    plot_waterfall(out_dir / "figures" / "fig02_processed_products.png", {
        "processed_bg": processed_background,
        "processed_sat": processed_starlink_only,
        "interaction": interaction_residual,
    }, ctx.freqs_hz)
    plot_delay_profiles(out_dir / "figures" / "fig03_delay_profiles.png", {
        "raw_starlink": raw_starlink,
        "processed_starlink": processed_starlink_only,
        "interaction": interaction_residual,
    }, ctx, cfg)

    full_report = {
        "experiment": cfg.get("experiment", {}),
        "background": {
            "source_format": ctx.source_format,
            "source_path": ctx.source_path,
            "shape": list(ctx.vis_tf.shape),
            "freq_min_mhz": float(ctx.freqs_hz.min() / 1e6),
            "freq_max_mhz": float(ctx.freqs_hz.max() / 1e6),
            "n_time": int(len(ctx.times_jd)),
            "n_freq": int(len(ctx.freqs_hz)),
            "baseline_enu_m": ctx.baseline_enu_m.tolist(),
            "baseline_m": float(np.linalg.norm(ctx.baseline_enu_m)),
            "weight_nonzero_fraction": float(np.mean(ctx.weights_tf > 0)),
            "metadata": ctx.metadata,
        },
        "starlink_injection": inj_report,
        "pipeline_background": pipe_bg,
        "pipeline_starlink_only": pipe_sat,
        "pipeline_dirty": pipe_dirty,
        "summary": summary,
        "dependency_status": {
            "skyfield": "ok" if skyfield_load is not None else SKYFIELD_ERROR,
            "scipy": "ok" if blackmanharris is not None else SCIPY_ERROR,
            "hera_sim_polybeam": "ok" if PolyBeam is not None else POLYBEAM_ERROR,
        },
        "non_claims": [
            "No internally generated foreground is used.",
            "No redundant calibration is implemented or claimed.",
            "No exact proprietary Starlink waveform reconstruction is claimed unless supplied as measured template.",
            "Flags are used as weights; flagged samples are not median-replaced.",
        ],
    }
    with open(out_dir / "reports" / "summary_report.json", "w", encoding="utf-8") as f:
        json.dump(full_report, f, indent=2, ensure_ascii=False)
    with open(out_dir / "config_used.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=True)

    md = [
        "# Near-field Starlink/UEMR injection on external HERA background",
        "",
        "## Critical interpretation",
        "",
        "- This run uses an externally provided background visibility. No toy foreground is created internally.",
        "- The Starlink/LEO delay is near-field: antenna-to-satellite range difference, not b·s/c.",
        "- The pipeline is a weighted delay/fringe-rate analysis, not redundant calibration.",
        "- The main contamination product is `interaction_residual = pipeline(background + starlink) - pipeline(background)`.",
        "",
        "## Key numbers",
        "",
        f"- interaction vs processed background: `{summary['interaction_vs_processed_background_db']:.3f}` dB",
        f"- processed starlink survival vs raw starlink: `{summary['processed_starlink_survival_vs_raw_starlink_db']:.3f}` dB",
        f"- interaction survival vs raw starlink: `{summary['interaction_survival_vs_raw_starlink_db']:.3f}` dB",
        f"- peak altitude: `{inj_report['peak_alt_deg']:.3f}` deg",
        f"- range min/max: `{inj_report['range_min_km']:.3f}` / `{inj_report['range_max_km']:.3f}` km",
        "",
        "## Outputs",
        "",
        "- `arrays/interaction_residual.npy`: main paper-facing residual product.",
        "- `tables/summary_metrics.csv`: weighted power and delay-excess metrics.",
        "- `tables/subspace_overlap.csv`: SVD subspace-overlap diagnostics.",
        "- `reports/summary_report.json`: machine-readable provenance report.",
    ]
    (out_dir / "reports" / "REPORT.md").write_text("\n".join(md), encoding="utf-8")

    print(json.dumps({
        "output_dir": str(out_dir),
        "interaction_vs_processed_background_db": summary["interaction_vs_processed_background_db"],
        "processed_starlink_survival_vs_raw_starlink_db": summary["processed_starlink_survival_vs_raw_starlink_db"],
        "emission_type": inj_report["emission_type"],
        "beam_model": inj_report["beam_model"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
