#!/usr/bin/env python3
"""
jones_polarization_engine.py
============================
Full Jones-matrix polarization projection engine for near-field LEO satellite
injection into radio interferometric visibilities.

Measurement equation (per baseline ij, time t, frequency ν):

    V_ij(t,ν) = J_i(t,ν) · C_src(t,ν) · J_j†(t,ν)

Shapes:
    J_i, J_j : (Nt, Nf, 2, 2) complex — antenna Jones matrix
    C_src     : (Nt, Nf, 2, 2) complex — source coherency (brightness) matrix
    V_ij      : (Nt, Nf, 2, 2) complex — [XX, XY; YX, YY]

Physical limitations (explicitly documented for paper Methods):
    - Starlink UEMR emission Jones matrix is UNKNOWN; C_src is modeled
      from Bassa et al. 2024 / Di Vruno et al. 2023 observations only.
    - PolyBeam is a primary-beam model; mutual coupling (M-terms) is absent.
    - D-terms (cross-feed leakage) are set to zero.
    - Ionospheric Faraday rotation is not modeled.
    - This engine operates on a single baseline (single ij pair).

Dependencies:
    numpy (required)
    hera_sim.beams.PolyBeam (optional; falls back to Gaussian approximation)
"""
from __future__ import annotations

import math
import warnings
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple

import numpy as np

C_M_PER_S = 299_792_458.0

# ---------------------------------------------------------------------------
# Try to import PolyBeam; fall back gracefully
# ---------------------------------------------------------------------------
try:
    from hera_sim.beams import PolyBeam as _PolyBeam  # type: ignore
    _POLYBEAM_OK = True
except Exception as _e:
    _PolyBeam = None
    _POLYBEAM_OK = False
    _POLYBEAM_ERROR = repr(_e)


# ===========================================================================
# 1. Antenna Jones Matrix Evaluator
# ===========================================================================

class JonesBeamEvaluator:
    """Extract the 2×2 Jones matrix J_i from an antenna beam model.

    For an ideal dual-feed antenna in the (X, Y) basis:

        J = [[E_xx(θ,φ,ν),  E_xy(θ,φ,ν)],
             [E_yx(θ,φ,ν),  E_yy(θ,φ,ν)]]

    Row p : feed polarization p receives  (p ∈ {X, Y})
    Col q : sky component q of the E-field (q ∈ {θ, φ})

    For a perfectly aligned, zero-leakage antenna:
        J = diag(E_x(θ,φ,ν), E_y(θ,φ,ν))   (diagonal Jones)

    PolyBeam returns the primary-beam E-field amplitude pattern.
    The off-diagonal D-terms are set to zero here.

    Parameters
    ----------
    mode : str
        "polybeam_fagnoni19" — hera_sim PolyBeam (recommended)
        "gaussian"           — symmetric Gaussian fallback (FWHM ≈ 10°/ν_GHz)
        "isotropic"          — J = I₂ everywhere (test only)
    """

    def __init__(self, mode: str = "polybeam_fagnoni19"):
        self.mode = mode.lower()
        self._beam = None
        self._meta: Dict[str, Any] = {"mode": self.mode, "d_terms_included": False}
        if self.mode == "polybeam_fagnoni19":
            if not _POLYBEAM_OK:
                warnings.warn(
                    f"hera_sim PolyBeam unavailable ({_POLYBEAM_ERROR}); "
                    "falling back to Gaussian beam.",
                    stacklevel=2,
                )
                self.mode = "gaussian"
            else:
                self._beam = _PolyBeam.like_fagnoni19()
                self._meta["beam_class"] = "hera_sim.beams.PolyBeam.like_fagnoni19"
                self._meta["reference"] = "Fagnoni et al. 2021 MNRAS"

    def evaluate(
        self,
        alt_deg: np.ndarray,   # shape (Nt,)
        az_deg:  np.ndarray,   # shape (Nt,)
        freqs_hz: np.ndarray,  # shape (Nf,)
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Compute J(t, ν) for a time-varying pointing direction.

        Returns
        -------
        J : ndarray, shape (Nt, Nf, 2, 2), complex
            J[t, f, 0, 0] = X-feed response to X-sky (E_xx)
            J[t, f, 0, 1] = X-feed response to Y-sky (E_xy = 0, no D-terms)
            J[t, f, 1, 0] = Y-feed response to X-sky (E_yx = 0, no D-terms)
            J[t, f, 1, 1] = Y-feed response to Y-sky (E_yy)
        meta : dict
            Provenance and limitation flags.
        """
        nt = len(alt_deg)
        nf = len(freqs_hz)
        J = np.zeros((nt, nf, 2, 2), dtype=complex)
        za_rad = np.deg2rad(90.0 - np.clip(alt_deg, 0.0, 90.0))
        az_rad = np.deg2rad(az_deg)

        if self.mode == "isotropic":
            J[..., 0, 0] = 1.0
            J[..., 1, 1] = 1.0
            return J, {**self._meta, "effective_mode": "isotropic_unity"}

        if self.mode == "gaussian":
            # Frequency-scaled FWHM: σ ≈ FWHM / (2√(2 ln 2))
            # HERA-like FWHM ≈ 10° at 150 MHz, scales as ν⁻¹
            fwhm_ref_deg = 10.0
            freq_ref_hz = 150e6
            for fi, f in enumerate(freqs_hz):
                fwhm_deg = fwhm_ref_deg * (freq_ref_hz / max(f, 1.0))
                sigma_rad = math.radians(fwhm_deg) / (2.0 * math.sqrt(2.0 * math.log(2.0)))
                amp = np.exp(-0.5 * (za_rad / sigma_rad) ** 2)
                J[:, fi, 0, 0] = amp.astype(complex)
                J[:, fi, 1, 1] = amp.astype(complex)
            return J, {**self._meta, "effective_mode": "gaussian_fallback",
                       "fwhm_ref_deg": fwhm_ref_deg, "freq_ref_mhz": freq_ref_hz / 1e6}

        # PolyBeam path
        try:
            ef = np.asarray(
                self._beam.efield_eval(az_array=az_rad, za_array=za_rad, freq_array=freqs_hz)
            )
        except Exception as exc:
            warnings.warn(f"PolyBeam.efield_eval failed ({exc}); falling back to Gaussian.", stacklevel=2)
            self.mode = "gaussian"
            return self.evaluate(alt_deg, az_deg, freqs_hz)

        # ef may be shape (Nfeed, Nt, Nf) or (Nt, Nf) depending on version.
        # We want E_x[t,f] and E_y[t,f].
        ef = np.asarray(ef)
        shape = ef.shape
        raw_max = float(np.nanmax(np.abs(ef)))
        raw_min = float(np.nanmin(np.abs(ef)))

        # Infer axes: look for (nt,) and (nf,) dimensions.
        t_axes = [i for i, s in enumerate(shape) if s == nt]
        f_axes = [i for i, s in enumerate(shape) if s == nf]

        if not t_axes or not f_axes:
            warnings.warn(
                f"Cannot infer PolyBeam axes from shape {shape} (nt={nt}, nf={nf}). "
                "Falling back to Gaussian.",
                stacklevel=2,
            )
            self.mode = "gaussian"
            return self.evaluate(alt_deg, az_deg, freqs_hz)

        # Move to canonical (Nt, Nf, ...) layout.
        ef_moved = np.moveaxis(ef, [t_axes[-1], f_axes[0]], [0, 1])  # (Nt, Nf, ...)
        # Average any remaining axes (feed dims, etc.) → (Nt, Nf)
        if ef_moved.ndim > 2:
            ef_moved = ef_moved.reshape(nt, nf, -1).mean(axis=2)

        # Clip unphysical excursions outside the fitted PolyBeam domain.
        ef_amp = np.clip(np.abs(ef_moved), 0.0, 1.0)
        ef_phase = np.angle(ef_moved)
        E_field = ef_amp * np.exp(1j * ef_phase)

        # Diagonal Jones: zero D-terms (cross-feed leakage)
        # J_xx ≈ J_yy for a symmetric dual-feed like HERA
        J[:, :, 0, 0] = E_field
        J[:, :, 1, 1] = E_field  # same beam for both feeds (HERA approximation)
        # J[..., 0, 1] = J[..., 1, 0] = 0  (no D-terms — already zero)

        meta = {
            **self._meta,
            "effective_mode": "polybeam_fagnoni19",
            "raw_efield_max": raw_max,
            "raw_efield_min": raw_min,
            "clipped_to_unit_efield": bool(raw_max > 1.0),
            "d_terms": "zero (diagonal Jones approximation)",
            "limitation": (
                "Off-diagonal D-terms (cross-feed leakage) are set to zero. "
                "Mutual coupling between HERA antennas is not modeled. "
                "Same E-field pattern assumed for X and Y feeds (symmetric beam)."
            ),
        }
        return J.astype(complex), meta


# ===========================================================================
# 2. Source Coherency (Brightness) Matrix
# ===========================================================================

class SourceCoherencyModel:
    """Model the 2×2 source coherency matrix C_src for Starlink UEMR.

    The coherency matrix relates to Stokes parameters as:
        C = [[I + Q,   U + iV],
             [U - iV,  I - Q ]]

    For an unpolarized source: C = I · [[1, 0], [0, 1]] (Stokes I only, Q=U=V=0)

    Physical note on Starlink UEMR:
        - Bassa et al. 2024 report anti-correlated XX/YY amplitudes in LOFAR data.
        - Di Vruno et al. 2023 do not provide a full Stokes decomposition.
        - The true satellite emission Jones matrix is unknown.
        - We parameterize C_src to be consistent with the reported anti-correlation
          while acknowledging this is not a full Stokes measurement.

    Methods
    -------
    unpolarized(S_tf)          : C = S · I₂ (safest assumption)
    anti_correlated(S_tf, ...) : Encode Bassa 2024 XX/YY anti-correlation via Stokes Q
    partially_polarized(S_tf, stokes_q_fraction)
                               : Generic partial linear polarization
    """

    @staticmethod
    def unpolarized(S_tf: np.ndarray) -> np.ndarray:
        """C_src = S(t,ν) · I₂

        Parameters
        ----------
        S_tf : array (Nt, Nf), real — Stokes I amplitude (Jy-like units)

        Returns
        -------
        C : array (Nt, Nf, 2, 2), complex
        """
        nt, nf = S_tf.shape
        C = np.zeros((nt, nf, 2, 2), dtype=complex)
        C[..., 0, 0] = S_tf
        C[..., 1, 1] = S_tf
        return C

    @staticmethod
    def anti_correlated(
        S_tf: np.ndarray,
        contrast: float = 0.35,
        phase_rad: float = 0.0,
    ) -> np.ndarray:
        """Encode Bassa et al. 2024 anti-correlated XX/YY as Stokes Q.

        Observation: V_XX ≠ V_YY with anti-correlation.
        In Stokes terms: if XX > YY then Q > 0.

        Model:
            I = S (total intensity)
            Q = contrast × S × sin(phase + π/2)   [varies slowly]
            U = V = 0 (no circular or 45° linear polarization assumed)

        This is a measurement-inspired parameterization, NOT a waveform reconstruction.
        The actual Stokes Q/U/V of Starlink UEMR has not been published.

        Parameters
        ----------
        S_tf     : array (Nt, Nf) — Stokes I amplitude
        contrast : float — fractional linear polarization (0=unpolarized, 1=fully)
                   Bassa 2024 implies |XX−YY|/(XX+YY) ~ 0.3–0.5 in some detections.
        phase_rad: float — initial phase of Q oscillation along time axis

        Returns
        -------
        C : array (Nt, Nf, 2, 2), complex
        """
        nt, nf = S_tf.shape
        t_norm = np.linspace(0, 2 * np.pi, nt)
        Q_t = contrast * np.sin(t_norm + phase_rad)  # slow oscillation (Nt,)
        Q_tf = Q_t[:, None] * S_tf  # (Nt, Nf)

        C = np.zeros((nt, nf, 2, 2), dtype=complex)
        C[..., 0, 0] = S_tf + Q_tf   # XX = I + Q
        C[..., 1, 1] = S_tf - Q_tf   # YY = I - Q
        # C[..., 0, 1] = U + iV = 0  (no U, V assumed)
        # C[..., 1, 0] = U - iV = 0
        return C

    @staticmethod
    def partially_polarized(
        S_tf: np.ndarray,
        stokes_q_fraction: float = 0.0,
        stokes_u_fraction: float = 0.0,
    ) -> np.ndarray:
        """Generic partial linear polarization.

        C = [[I+Q,   U],
             [U,     I-Q]]

        (Stokes V = 0, no circular polarization)

        Parameters
        ----------
        S_tf              : (Nt, Nf) — Stokes I
        stokes_q_fraction : Q / I  (fraction of total power in Q)
        stokes_u_fraction : U / I

        Returns
        -------
        C : (Nt, Nf, 2, 2), complex
        """
        nt, nf = S_tf.shape
        Q = stokes_q_fraction * S_tf
        U = stokes_u_fraction * S_tf
        C = np.zeros((nt, nf, 2, 2), dtype=complex)
        C[..., 0, 0] = S_tf + Q   # I + Q
        C[..., 0, 1] = U
        C[..., 1, 0] = U
        C[..., 1, 1] = S_tf - Q   # I - Q
        return C


# ===========================================================================
# 3. Jones Visibility Projector
# ===========================================================================

class JonesProjector:
    """Apply the interferometric measurement equation:

        V_ij(t,ν) = J_i(t,ν) · C_src(t,ν) · J_j†(t,ν)

    All arrays are shape (Nt, Nf, 2, 2).
    The operation is:
        V[t,f] = J_i[t,f] @ C[t,f] @ J_j[t,f].conj().T

    For a single selected polarization product (e.g., ee ≈ XX):
        V_XX[t,f] = V[t,f,0,0]
        V_YY[t,f] = V[t,f,1,1]
        V_XY[t,f] = V[t,f,0,1]
        V_YX[t,f] = V[t,f,1,0]
    """

    @staticmethod
    def project(
        J_i: np.ndarray,   # (Nt, Nf, 2, 2) complex
        J_j: np.ndarray,   # (Nt, Nf, 2, 2) complex
        C_src: np.ndarray, # (Nt, Nf, 2, 2) complex
    ) -> np.ndarray:
        """Compute V = J_i · C_src · J_j†  using batched matrix multiply.

        Returns
        -------
        V : (Nt, Nf, 2, 2) complex — full 2×2 visibility matrix
        """
        # J_j† = J_j.conj().swapaxes(-2, -1)
        J_j_dag = np.conj(J_j).swapaxes(-2, -1)
        # numpy matmul broadcasts over leading dims: (Nt, Nf, 2, 2) @ (Nt, Nf, 2, 2)
        V = np.matmul(np.matmul(J_i, C_src), J_j_dag)
        return V.astype(complex)

    @staticmethod
    def extract_pol(
        V_full: np.ndarray,   # (Nt, Nf, 2, 2)
        pol: str = "xx",
    ) -> np.ndarray:
        """Extract a single polarization product from the 2×2 visibility matrix.

        pol : one of "xx", "xy", "yx", "yy", "ee", "nn", "en", "ne"
              HERA 'ee' maps to XX (feed 0), 'nn' to YY (feed 1).
        """
        pol = pol.lower().strip()
        _map = {
            "xx": (0, 0), "ee": (0, 0), "x": (0, 0),
            "xy": (0, 1), "en": (0, 1),
            "yx": (1, 0), "ne": (1, 0),
            "yy": (1, 1), "nn": (1, 1), "y": (1, 1),
        }
        if pol not in _map:
            raise ValueError(f"Unknown polarization '{pol}'. Choose from {list(_map.keys())}.")
        i, j = _map[pol]
        return V_full[..., i, j]  # (Nt, Nf), complex


# ===========================================================================
# 4. Stokes Extractor
# ===========================================================================

class StokesExtractor:
    """Convert 2×2 visibility matrix to Stokes parameters.

    For a calibrated interferometer:
        I = (XX + YY) / 2
        Q = (XX - YY) / 2
        U = (XY + YX) / 2
        V = (XY - YX) / (2i)

    Note: This is the ideal (D-term free) conversion.
    In the presence of leakage, I ↔ Q, U, V cross-contaminate.
    """

    @staticmethod
    def to_stokes(V_full: np.ndarray) -> Dict[str, np.ndarray]:
        """Compute all four Stokes from (Nt, Nf, 2, 2) visibility.

        Returns dict with keys 'I', 'Q', 'U', 'V', each shape (Nt, Nf) complex.
        """
        XX = V_full[..., 0, 0]
        XY = V_full[..., 0, 1]
        YX = V_full[..., 1, 0]
        YY = V_full[..., 1, 1]
        return {
            "I": 0.5 * (XX + YY),
            "Q": 0.5 * (XX - YY),
            "U": 0.5 * (XY + YX),
            "V": 0.5 * (XY - YX) / 1j,
        }


# ===========================================================================
# 5. Full Pipeline Engine
# ===========================================================================

@dataclass
class JonesPolarizationResult:
    """Output of JonesPolarizationEngine.project_satellite()"""
    # Selected single-pol visibility for use in main pipeline
    vis_selected_pol: np.ndarray      # (Nt, Nf) complex
    # Full 2×2 visibility matrix (for paper diagnostics / Stokes analysis)
    vis_full_jones: np.ndarray        # (Nt, Nf, 2, 2) complex
    # Jones matrices used
    J_i: np.ndarray                   # (Nt, Nf, 2, 2)
    J_j: np.ndarray                   # (Nt, Nf, 2, 2)  — same as J_i for same beam model
    # Source coherency used
    C_src: np.ndarray                 # (Nt, Nf, 2, 2)
    # Stokes diagnostics
    stokes: Dict[str, np.ndarray]     # I, Q, U, V each (Nt, Nf)
    # Metadata and non-claims
    meta: Dict[str, Any] = field(default_factory=dict)


class JonesPolarizationEngine:
    """Orchestrates full Jones-matrix polarization projection for a satellite pass.

    Usage
    -----
    engine = JonesPolarizationEngine(beam_mode="polybeam_fagnoni19")
    result = engine.project_satellite(
        S_tf       = amplitude_model,    # (Nt, Nf) flux density
        alt_deg    = track["alt_deg"],
        az_deg     = track["az_deg"],
        freqs_hz   = ctx.freqs_hz,
        source_coherency_model = "anti_correlated",
        contrast   = 0.35,
        selected_pol = "ee",
    )
    vis_to_inject = result.vis_selected_pol  # drop-in replacement for scalar vis

    Non-claims (logged in result.meta)
    ------------------------------------
    - C_src is modeled from literature observations, not a satellite waveform measurement.
    - D-terms are zero (diagonal Jones approximation).
    - Both antennas assumed to have identical beam patterns (J_i ≈ J_j).
    - Mutual coupling, ionospheric Faraday rotation, and RFI direction-of-arrival
      structure within the beam are all neglected.

    Paper Methods text template
    ---------------------------
    "Full Jones-matrix polarization projection was applied following the measurement
    equation V_ij = J_i · C_src · J_j† (e.g., Hamaker et al. 1996; Smirnov 2011).
    The antenna response J was evaluated using the hera_sim PolyBeam model
    (Fagnoni et al. 2021). The satellite emission coherency matrix C_src was modeled
    as [unpolarized / anti-correlated with contrast δ based on Bassa et al. 2024].
    Cross-feed leakage (D-terms) and ionospheric Faraday rotation were neglected.
    A full polarization propagation including D-terms is left for future work."
    """

    NON_CLAIMS = [
        "No proprietary Starlink waveform reconstruction is claimed.",
        "Source coherency C_src is a literature-parameterized model, not a direct measurement.",
        "D-terms (cross-feed leakage) are set to zero (diagonal Jones approximation).",
        "J_i = J_j assumed (identical beam pattern for both antennas of the baseline).",
        "Mutual coupling, M-terms, and ionospheric Faraday rotation are not modeled.",
        "Full cross-polarization leakage analysis is left for future work.",
    ]

    def __init__(self, beam_mode: str = "polybeam_fagnoni19"):
        self.beam_evaluator = JonesBeamEvaluator(mode=beam_mode)
        self.projector = JonesProjector()
        self.stokes_extractor = StokesExtractor()

    def project_satellite(
        self,
        S_tf: np.ndarray,           # (Nt, Nf) real — source amplitude before beam
        alt_deg: np.ndarray,        # (Nt,) — satellite elevation
        az_deg: np.ndarray,         # (Nt,) — satellite azimuth
        freqs_hz: np.ndarray,       # (Nf,)
        source_coherency_model: str = "unpolarized",
        contrast: float = 0.35,     # for anti_correlated model
        phase_rad: float = 0.0,     # for anti_correlated model
        stokes_q_fraction: float = 0.0,
        stokes_u_fraction: float = 0.0,
        selected_pol: str = "ee",
    ) -> JonesPolarizationResult:
        """Apply V_ij = J_i · C_src · J_j† to satellite emission.

        Parameters
        ----------
        S_tf     : (Nt, Nf) — amplitude envelope from build_starlink_visibility
                   (flux_ref × range_att × attenuation × spec_amp × duty_t ...)
                   This is the REAL amplitude; phase will be added externally via
                   exp(-2πiν τ_ij).
        source_coherency_model : str
            "unpolarized"         — C = S · I₂  (safest; default)
            "anti_correlated"     — Bassa 2024 inspired XX/YY anti-correlation
            "partially_polarized" — generic Stokes Q/U injection
        """
        S_tf = np.asarray(S_tf, dtype=float)
        nt, nf = S_tf.shape

        # 1. Evaluate antenna Jones matrix
        J, beam_meta = self.beam_evaluator.evaluate(alt_deg, az_deg, freqs_hz)
        # Both antennas use the same beam model (J_i ≈ J_j for HERA)
        J_i = J
        J_j = J  # limitation: assumes identical beam patterns

        # 2. Build source coherency matrix
        model = source_coherency_model.lower()
        if model == "unpolarized":
            C_src = SourceCoherencyModel.unpolarized(S_tf)
            c_meta = {"model": "unpolarized", "stokes": "I_only"}
        elif model == "anti_correlated":
            C_src = SourceCoherencyModel.anti_correlated(S_tf, contrast, phase_rad)
            c_meta = {
                "model": "anti_correlated",
                "contrast": contrast,
                "phase_rad": phase_rad,
                "reference": "Bassa et al. 2024 A&A (LOFAR v2-Mini UEMR)",
                "limitation": (
                    "Q parameterized from reported XX/YY anti-correlation. "
                    "True Stokes Q/U/V of Starlink UEMR has not been published."
                ),
            }
        elif model == "partially_polarized":
            C_src = SourceCoherencyModel.partially_polarized(S_tf, stokes_q_fraction, stokes_u_fraction)
            c_meta = {
                "model": "partially_polarized",
                "stokes_q_fraction": stokes_q_fraction,
                "stokes_u_fraction": stokes_u_fraction,
            }
        else:
            raise ValueError(
                f"Unknown source_coherency_model='{source_coherency_model}'. "
                "Choose: unpolarized, anti_correlated, partially_polarized."
            )

        # 3. Apply measurement equation: V = J_i · C_src · J_j†
        V_full = self.projector.project(J_i, J_j, C_src)  # (Nt, Nf, 2, 2)

        # 4. Extract selected polarization product
        vis_selected = self.projector.extract_pol(V_full, selected_pol)  # (Nt, Nf)

        # 5. Stokes diagnostics
        stokes = self.stokes_extractor.to_stokes(V_full)

        # 6. Cross-polarization leakage summary (for paper Table/Methods)
        leakage_XY = float(np.nanmean(np.abs(V_full[..., 0, 1]) ** 2))
        leakage_YX = float(np.nanmean(np.abs(V_full[..., 1, 0]) ** 2))
        power_XX   = float(np.nanmean(np.abs(V_full[..., 0, 0]) ** 2))
        power_YY   = float(np.nanmean(np.abs(V_full[..., 1, 1]) ** 2))
        leakage_fraction = (leakage_XY + leakage_YX) / max(power_XX + power_YY, 1e-30)

        meta = {
            "jones_engine_version": "1.0",
            "selected_pol": selected_pol,
            "beam_meta": beam_meta,
            "source_coherency": c_meta,
            "non_claims": self.NON_CLAIMS,
            "diagnostics": {
                "power_XX": power_XX,
                "power_YY": power_YY,
                "leakage_XY_power": leakage_XY,
                "leakage_YX_power": leakage_YX,
                "cross_pol_leakage_fraction": leakage_fraction,
                "stokes_I_mean": float(np.nanmean(np.abs(stokes["I"]))),
                "stokes_Q_mean": float(np.nanmean(np.abs(stokes["Q"]))),
                "stokes_Q_over_I_mean": float(
                    np.nanmean(np.abs(stokes["Q"]))
                    / max(np.nanmean(np.abs(stokes["I"])), 1e-30)
                ),
                "vis_selected_pol_peak_abs": float(np.nanmax(np.abs(vis_selected))),
            },
            "paper_limitation": (
                "Full cross-polarization leakage analysis (D-terms, ionospheric Faraday "
                "rotation, mutual coupling) is left for future work. The current implementation "
                "uses a diagonal Jones approximation with parameterized source coherency."
            ),
        }

        return JonesPolarizationResult(
            vis_selected_pol=vis_selected,
            vis_full_jones=V_full,
            J_i=J_i,
            J_j=J_j,
            C_src=C_src,
            stokes=stokes,
            meta=meta,
        )


# ===========================================================================
# 6. Integration helper for run_nearfield_starlink_on_background.py
# ===========================================================================

def build_jones_starlink_visibility(
    amp_tf: np.ndarray,      # (Nt, Nf) — amplitude envelope from main script
    tau_s: np.ndarray,       # (Nt,) — near-field delay [seconds]
    freqs_hz: np.ndarray,    # (Nf,)
    alt_deg: np.ndarray,     # (Nt,)
    az_deg: np.ndarray,      # (Nt,)
    source_coherency_model: str = "unpolarized",
    beam_mode: str = "polybeam_fagnoni19",
    contrast: float = 0.35,
    selected_pol: str = "ee",
) -> Tuple[np.ndarray, JonesPolarizationResult]:
    """Drop-in replacement for the scalar visibility builder in the main script.

    Replaces:
        vis = amp_tf * phase

    With:
        vis = JonesPolarizationEngine.project_satellite(amp_tf, ...) * phase

    The near-field phase term exp(-2πiν τ_ij(t)) is applied AFTER Jones projection
    because it encodes the geometric delay of the interferometer, not the source polarization.

    Returns
    -------
    vis : (Nt, Nf) complex — selected-pol visibility ready for injection
    result : JonesPolarizationResult — full diagnostics for paper reporting
    """
    engine = JonesPolarizationEngine(beam_mode=beam_mode)
    result = engine.project_satellite(
        S_tf=amp_tf,
        alt_deg=alt_deg,
        az_deg=az_deg,
        freqs_hz=freqs_hz,
        source_coherency_model=source_coherency_model,
        contrast=contrast,
        selected_pol=selected_pol,
    )
    # Apply near-field interferometric phase (geometry, not polarization)
    phase = np.exp(-2j * np.pi * tau_s[:, None] * freqs_hz[None, :])
    vis = result.vis_selected_pol * phase
    return vis, result


# ===========================================================================
# 7. Simulation-vs-real comparison (honest limitation table)
# ===========================================================================

SIMULATION_LIMITATIONS = {
    "system_noise": {
        "simulation": "Gaussian thermal noise, SEFD from config",
        "real_data": "ADC nonlinearity, board-to-board timing jitter, RFI from electronics",
        "impact": "Noise floor shape differs; flagging thresholds need in-situ tuning",
        "mitigation": "Use measured noise diode calibration; fit SEFD from bright source transits",
    },
    "near_field_delay": {
        "simulation": "Geometric (ENU); no ionospheric extra delay",
        "real_data": "Geometric + TEC-dependent ionospheric delay (ΔΔτ ~ 1-10 ns at 150 MHz)",
        "impact": "Residual delay after fringe correction; could shift delay-domain leakage",
        "mitigation": "GPS TEC maps (IONEX); or GPSIon from pyuvdata",
    },
    "jones_beam": {
        "simulation": "PolyBeam model (Fagnoni 2021); D-terms=0; no mutual coupling",
        "real_data": "Full primary beam from in-situ holography (Berber et al. 2023)",
        "impact": "~5-15% beam amplitude error at large zenith angles; sidelobes wrong",
        "mitigation": "Replace J with measured FEE beam or holographic map",
    },
    "satellite_polarization": {
        "simulation": "C_src modeled from Bassa 2024 (contrast~0.35); true Stokes unknown",
        "real_data": "Full Stokes decomposition requires 4 cross-correlations simultaneously",
        "impact": "XX/YY contamination ratio uncertain; leakage into Stokes U,V unknown",
        "mitigation": "Simultaneous EE+NN+EN+NE observation; Stokes decomposition per satellite",
    },
    "source_catalog": {
        "simulation": "Background from hera_sim/pyuvsim; no real sky catalog validation",
        "real_data": "Real foreground deviates from model at ~1-10% level",
        "impact": "Interaction residual may be confused with foreground model errors",
        "mitigation": "Use real UVH5 as background; apply null test across nights",
    },
}


def print_limitation_table() -> None:
    """Print a human-readable table of simulation vs real-data limitations."""
    print("\n" + "=" * 80)
    print("SIMULATION vs REAL DATA: Limitation Table")
    print("=" * 80)
    for component, info in SIMULATION_LIMITATIONS.items():
        print(f"\n[{component.upper()}]")
        for k, v in info.items():
            print(f"  {k:12s}: {v}")
    print("=" * 80 + "\n")


# ===========================================================================
# Self-test / smoke test
# ===========================================================================

def _smoke_test() -> None:
    """Quick sanity check: shapes and physical consistency."""
    print("Running JonesPolarizationEngine smoke test...")
    rng = np.random.default_rng(42)
    Nt, Nf = 96, 64
    freqs = np.linspace(110e6, 190e6, Nf)
    alt = np.linspace(20.0, 60.0, Nt)
    az  = np.linspace(90.0, 270.0, Nt)
    S_tf = np.abs(rng.normal(100.0, 10.0, (Nt, Nf)))  # Jy-like amplitude

    engine = JonesPolarizationEngine(beam_mode="gaussian")

    for model in ["unpolarized", "anti_correlated", "partially_polarized"]:
        result = engine.project_satellite(
            S_tf=S_tf,
            alt_deg=alt,
            az_deg=az,
            freqs_hz=freqs,
            source_coherency_model=model,
            selected_pol="ee",
        )
        assert result.vis_selected_pol.shape == (Nt, Nf), f"Shape mismatch for {model}"
        assert result.vis_full_jones.shape == (Nt, Nf, 2, 2)
        assert np.all(np.isfinite(result.vis_selected_pol))
        # Unpolarized: Q should be zero; anti_correlated: Q should be non-zero
        Q_mean = float(np.nanmean(np.abs(result.stokes["Q"])))
        if model == "unpolarized":
            assert Q_mean < 1e-10, f"Unpolarized model has Q={Q_mean:.2e}"
        print(f"  [{model}] OK — vis peak={np.nanmax(np.abs(result.vis_selected_pol)):.3e}  "
              f"Q/I={result.meta['diagnostics']['stokes_Q_over_I_mean']:.3f}")

    print("All smoke tests passed.\n")
    print_limitation_table()


if __name__ == "__main__":
    _smoke_test()
