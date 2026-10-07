"""Synthetic InSAR deformation generator for the cross-modality experiment.

Physical model (fully specified, analytic, no external data):
  * a Mogi (1958) point pressure source in a homogeneous elastic half-space,
        u_r = C r / (r^2 + d^2)^(3/2),  u_z = C d / (r^2 + d^2)^(3/2),
    with C = (1-nu)/pi * dV; displacement is LINEAR in dV, so inversion only
    grid-searches (x0, y0, d) and solves dV by least squares;
  * single-track (descending) LOS projection from incidence/heading angles;
  * a regular Grid2D geometry carrying multi-epoch LOS displacement and
    coherence NDArrays (millimetres);
  * tropospheric delay as a long-wavelength, power-law random field that is
    approximately TEMPORALLY INDEPENDENT between epochs (the dominant false
    anomaly: a single-epoch atmospheric bump looks like local deformation);
  * decorrelation as spatially clustered low-coherence patches/voids;
  * a small static long-wavelength DEM residual;
  * per-pixel phase noise from the Cramer-Rao coherence relation.

As in the microseismic generator, quality is estimated FROM THE DATA (the
coherence product and the LOS stack); ground truth is used only for scoring.
Deformation-free scenes use the SAME noise-floor distribution as deformation
scenes at each coherence level (no truth leakage).
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, Optional, Tuple

import numpy as np
from scipy.ndimage import gaussian_filter

from gpjson41.builder import GPJSONBuilder
from gpjson41.policies_insar import (
    COH_GATE,
    phase_sigma_mm,
    quality_from_coherence,
)


@dataclass
class InSARConfig:
    nx: int = 64
    ny: int = 64
    spacing_m: float = 30.0
    n_epochs: int = 16
    dt_days: float = 12.0            # revisit interval
    nu: float = 0.25                 # Poisson ratio
    incidence_deg: float = 34.0      # off-vertical
    heading_azimuth_deg: float = 102.0   # LOS horizontal azimuth, clockwise from N
    # source prior (local mine grid, metres); depth/position kept within the
    # field of view so the Mogi anomaly core is resolved without edge clipping
    xlim: Tuple[float, float] = (-450.0, 450.0)
    ylim: Tuple[float, float] = (-450.0, 450.0)
    depth_range: Tuple[float, float] = (300.0, 1100.0)
    peak_los_range_mm: Tuple[float, float] = (12.0, 40.0)
    # noise model
    atm_corr_m: float = 1200.0       # long-wavelength correlation length
    atm_beta: float = 11.0 / 3.0     # power-law spectral exponent
    atm_rms_mm_at_085: float = 3.0
    atm_rms_mm_at_030: float = 16.0
    dem_residual_mm: float = 2.0     # static long-wavelength residual
    dem_corr_m: float = 1200.0
    coh_short_sigma: float = 0.14    # pixel-scale coherence scatter
    coh_short_corr_m: float = 120.0
    void_corr_m: float = 700.0       # decorrelation patch correlation length
    # stable-scatterer "islands" (corner reflectors / PS on the monitored
    # slope): the source is sited on an island, mirroring real InSAR targeting.
    # Deformation-free scenes carry the same random islands (no truth leak).
    n_islands: int = 2
    island_radius_m: float = 320.0
    island_boost: float = 0.40
    # intent defaults (regional survey)
    goal: str = "deformation_trend_mapping"
    latency_budget_s: float = 86400.0
    priority: str = "normal"
    crs: str = "local:mine_grid"
    inline_limit_bytes: int = 8 * 1024 * 1024

    @property
    def span_years(self) -> float:
        return (self.n_epochs - 1) * self.dt_days / 365.25


# --------------------------------------------------------------------------
# Geometry and forward model
# --------------------------------------------------------------------------
def grid_coordinates(cfg: InSARConfig):
    x = (np.arange(cfg.nx) - (cfg.nx - 1) / 2.0) * cfg.spacing_m
    y = (np.arange(cfg.ny) - (cfg.ny - 1) / 2.0) * cfg.spacing_m
    X, Y = np.meshgrid(x, y)
    return x, y, X, Y


def los_unit_vector(cfg: InSARConfig) -> np.ndarray:
    """LOS unit vector [east, north, up] (motion toward satellite positive)."""
    th = math.radians(cfg.incidence_deg)
    az = math.radians(cfg.heading_azimuth_deg)
    return np.array([math.sin(th) * math.sin(az),
                     math.sin(th) * math.cos(az),
                     math.cos(th)])


def mogi_displacement(X, Y, sx, sy, depth, c_dv):
    """Surface displacement (metres) of a Mogi point source.

    c_dv = (1-nu)/pi * dV  (m^3). Returns ux(east), uy(north), uz(up) in m.
    """
    dx = X - sx
    dy = Y - sy
    r3 = (dx ** 2 + dy ** 2 + depth ** 2) ** 1.5
    ux = c_dv * dx / r3
    uy = c_dv * dy / r3
    uz = c_dv * depth / r3
    return ux, uy, uz


def c_dv_from_peak_los(cfg: InSARConfig, depth: float, peak_los_mm: float,
                       n_los: np.ndarray) -> float:
    """Solve c_dv (m^3) that yields the target peak LOS (above the source)."""
    # at r=0 only vertical motion: u_z(0) = c_dv / depth^2, LOS = n_up * u_z
    peak_m = peak_los_mm / 1000.0
    return peak_m * depth ** 2 / n_los[2]


# --------------------------------------------------------------------------
# Random fields
# --------------------------------------------------------------------------
def _power_law_field(ny, nx, dx, corr_m, beta, rng) -> np.ndarray:
    """Zero-mean, unit-std 2-D field with a power-law spectrum (~ k^-beta)."""
    kx = np.fft.fftfreq(nx, d=dx)
    ky = np.fft.fftfreq(ny, d=dx)
    KX, KY = np.meshgrid(kx, ky)
    k0 = 1.0 / corr_m
    white = rng.standard_normal((ny, nx))
    fw = np.fft.fft2(white)
    H = (KX ** 2 + KY ** 2 + k0 ** 2) ** (-beta / 4.0)
    H[0, 0] = 0.0
    f = np.real(np.fft.ifft2(fw * H))
    return f / (f.std() + 1e-12)


def _smooth_field(ny, nx, corr_m, dx, rng) -> np.ndarray:
    """Zero-mean, unit-std spatially smoothed Gaussian field."""
    f = rng.standard_normal((ny, nx))
    sigma_px = max(0.5, corr_m / (2.0 * dx))
    f = gaussian_filter(f, sigma=sigma_px, mode="reflect")
    return f / (f.std() + 1e-12)


def atm_rms_mm(mean_gamma: float, cfg: InSARConfig) -> float:
    """Atmospheric amplitude grows as coherence falls (linear, documented)."""
    g0, g1 = 0.85, 0.30
    a0, a1 = cfg.atm_rms_mm_at_085, cfg.atm_rms_mm_at_030
    g = np.clip(mean_gamma, g1, g0)
    return float(a0 + (a1 - a0) * (g0 - g) / (g0 - g1))


def _void_fraction(mean_gamma: float) -> float:
    """Fraction of pixels in clustered decorrelation voids vs mean coherence."""
    return float(np.clip(0.02 + 1.0 * (0.85 - mean_gamma) ** 1.3, 0.0, 0.38))


def make_islands(cfg: InSARConfig, rng: np.random.Generator):
    """Random stable-scatterer islands (centres, radii, coherence boost)."""
    islands = []
    for _ in range(cfg.n_islands):
        cx = float(rng.uniform(*cfg.xlim))
        cy = float(rng.uniform(*cfg.ylim))
        radius = float(rng.uniform(0.7, 1.2) * cfg.island_radius_m)
        boost = float(rng.uniform(cfg.island_boost, cfg.island_boost + 0.12))
        islands.append((cx, cy, radius, boost))
    return islands


def coherence_stack(cfg: InSARConfig, mean_gamma: float,
                    rng: np.random.Generator, islands=()) -> np.ndarray:
    """Build a (T, Ny, Nx) coherence cube with clustered low-coherence voids
    and stable-scatterer islands."""
    _, _, X, Y = grid_coordinates(cfg)
    ny, nx, T = cfg.ny, cfg.nx, cfg.n_epochs
    # spatial skeleton: clustered voids + short-scale scatter (stable in time)
    patch = _smooth_field(ny, nx, cfg.void_corr_m, cfg.spacing_m, rng)
    f_void = _void_fraction(mean_gamma)
    thresh = np.quantile(patch, f_void)
    void_mask = patch < thresh
    short = _smooth_field(ny, nx, cfg.coh_short_corr_m, cfg.spacing_m, rng)
    base = mean_gamma + cfg.coh_short_sigma * short
    g0 = np.where(
        void_mask,
        rng.uniform(0.05, COH_GATE - 0.05, size=(ny, nx)),
        np.clip(base, 0.05, 0.99),
    )
    # stable-scatterer islands raise coherence even inside poor regions
    for cx, cy, radius, boost in islands:
        bump = boost * np.exp(-((X - cx) ** 2 + (Y - cy) ** 2)
                              / (2.0 * radius ** 2))
        g0 = g0 + bump
    g0 = np.clip(g0, 0.05, 0.99)
    # small per-epoch fluctuation around the stable skeleton
    stack = np.empty((T, ny, nx), dtype=np.float64)
    for t in range(T):
        fl = rng.normal(0.0, 0.03, size=(ny, nx))
        stack[t] = np.clip(g0 + fl, 0.05, 0.99)
    return stack


# --------------------------------------------------------------------------
# Data-derived quality (ground truth never used here)
# --------------------------------------------------------------------------
def estimate_atm_rms_mm(los_mm: np.ndarray, cfg: InSARConfig,
                        coh: "np.ndarray | None" = None) -> float:
    """Estimate single-epoch atmospheric RMS from adjacent-epoch differences.

    Differences remove the steady deformation; their variance carries two
    independent atmospheric fields, so divide by sqrt(2). The coherence-based
    phase-noise variance is subtracted when the coherence cube is available.
    """
    diff = los_mm[1:] - los_mm[:-1]
    var = float(np.mean(diff ** 2))
    if coh is not None:
        noise_var = 2.0 * float(np.mean(phase_sigma_mm(coh) ** 2))
        var = max(var - noise_var, 1e-6)
    return math.sqrt(var / 2.0)


def estimate_dem_residual_mm(los_mm: np.ndarray, cfg: InSARConfig) -> float:
    """Static long-wavelength residual: low-pass of the temporal-mean image."""
    mean_img = los_mm.mean(axis=0)
    lp = gaussian_filter(mean_img, sigma=8.0, mode="reflect")
    return float(lp.std())


# --------------------------------------------------------------------------
# Scene generation
# --------------------------------------------------------------------------
def generate_scene(
    cfg: InSARConfig,
    mean_coherence: float,
    has_source: bool,
    rng: np.random.Generator,
    scene_index: int = 0,
    goal: Optional[str] = None,
    priority: Optional[str] = None,
    latency_budget_s: Optional[float] = None,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Generate one InSAR scene (T-epoch stack) as a GPJSON 4.1 Feature."""
    goal = cfg.goal if goal is None else goal
    priority = cfg.priority if priority is None else priority
    latency_budget_s = (cfg.latency_budget_s if latency_budget_s is None
                        else latency_budget_s)

    x, y, X, Y = grid_coordinates(cfg)
    n_los = los_unit_vector(cfg)
    T = cfg.n_epochs

    # stable-scatterer islands exist in BOTH source and source-free scenes
    islands = make_islands(cfg, rng)

    # --- source truth (sited on the first stable island) -----------------
    if has_source:
        sx, sy = float(islands[0][0]), float(islands[0][1])
        depth = float(rng.uniform(*cfg.depth_range))
        peak = float(rng.uniform(*cfg.peak_los_range_mm))
        c_dv = c_dv_from_peak_los(cfg, depth, peak, n_los)
        dv_m3 = c_dv * math.pi / (1.0 - cfg.nu)
        ux, uy, uz = mogi_displacement(X, Y, sx, sy, depth, c_dv)
        defo_final_mm = 1000.0 * (n_los[0] * ux + n_los[1] * uy
                                  + n_los[2] * uz)
    else:
        sx = sy = depth = dv_m3 = peak = None
        defo_final_mm = np.zeros((cfg.ny, cfg.nx), dtype=np.float64)

    # --- coherence and phase noise --------------------------------------
    coh = coherence_stack(cfg, mean_coherence, rng, islands)
    phase_sigma = phase_sigma_mm(coh)   # (T,Ny,Nx) mm

    # --- common noise floor (identical distribution with/without source) -
    a_atm = atm_rms_mm(mean_coherence, cfg)
    dem_field = cfg.dem_residual_mm * _power_law_field(
        cfg.ny, cfg.nx, cfg.spacing_m, cfg.dem_corr_m, cfg.atm_beta, rng)
    los = np.empty((T, cfg.ny, cfg.nx), dtype=np.float64)
    epoch_frac = np.linspace(0.0, 1.0, T)
    for t in range(T):
        atm = a_atm * _power_law_field(
            cfg.ny, cfg.nx, cfg.spacing_m, cfg.atm_corr_m, cfg.atm_beta, rng)
        noise = rng.normal(0.0, 1.0, size=(cfg.ny, cfg.nx)) * phase_sigma[t]
        los[t] = epoch_frac[t] * defo_final_mm + atm + dem_field + noise

    # --- data-derived quality -------------------------------------------
    coverage = float(np.mean(coh >= COH_GATE))
    mean_gamma_est = float(coh.mean())
    qmap = quality_from_coherence(mean_gamma_est, coverage)
    atm_est = estimate_atm_rms_mm(los, cfg, coh)
    dem_est = estimate_dem_residual_mm(los, cfg)

    # --- GPJSON 4.1 Feature ---------------------------------------------
    t0 = datetime(2025, 6, 1, 0, 0, 0, tzinfo=timezone.utc) + timedelta(
        days=scene_index * cfg.n_epochs * cfg.dt_days)
    fid = f"insar_scene_{scene_index:04d}"
    origin = [float(x[0]), float(y[0])]
    grid2d = {
        "type": "Grid2D",
        "origin": origin,
        "spacing": [cfg.spacing_m, cfg.spacing_m],
        "shape": [cfg.ny, cfg.nx],
        "axes": ["x_m", "y_m"],
    }

    b = GPJSONBuilder("insar.deformation", fid, crs=cfg.crs)
    b.set_geometry(grid2d)
    b.set_source("synth_insar_node@repro")
    b.set_timestamp(t0.strftime("%Y-%m-%dT%H:%M:%SZ"))
    b.add_quality(
        completeness=qmap["completeness"],
        outlier_rate=qmap["outlierRate"],
        confidence=qmap["confidence"],
        validity=qmap["validity"],
        uncertainty={"type": "gaussian",
                     "sigmaLosMm": "per-pixel, sigma ~ sqrt(1-g^2)/g",
                     "meanCoherence": qmap["meanCoherence"]},
    )
    b.set_pipeline(
        event_time=t0.strftime("%Y-%m-%dT%H:%M:%SZ"),
        latency_budget=latency_budget_s,
        qos={"priority": priority, "deliveryGuarantee": "at_least_once"},
        stage={"level": "raw", "readyForFusion": False},
        health={
            "sensorStatus": "degraded" if qmap["confidence"] < 0.6 else "OK",
            "meanCoherence": qmap["meanCoherence"],
            "coverage": qmap["completeness"],
            "atmPhaseRmsMm": atm_est,
            "demResidualMm": dem_est,
        },
        transport={"protocol": "https", "topic": "insar/desc/stack"},
    )
    b.set_intent(
        goal=goal,
        outcomes=["deformation_rate_map", "anomaly_alert"],
        constraints={"latencyBudget_s": latency_budget_s,
                     "accuracyClass":
                         "high_fidelity" if priority == "high" else "standard"},
    )
    b.add_property("n_epochs", cfg.n_epochs)
    b.add_property("dt_days", cfg.dt_days)
    b.add_property("incidence_deg", cfg.incidence_deg)
    b.add_property("heading_azimuth_deg", cfg.heading_azimuth_deg)
    b.add_dataset_ndarray(
        "los_displacement", los.astype(np.float32),
        axes=["epoch", "y", "x"], units="mm",
        inline_limit_bytes=cfg.inline_limit_bytes)
    b.add_dataset_ndarray(
        "coherence", coh.astype(np.float32),
        axes=["epoch", "y", "x"], units="dimensionless",
        inline_limit_bytes=cfg.inline_limit_bytes)
    b.add_dataset_ndarray(
        "los_unit_vector", n_los.astype(np.float32),
        axes=["component(east,north,up)"], units="dimensionless",
        inline_limit_bytes=cfg.inline_limit_bytes)
    feature = b.build()

    rate_true = defo_final_mm / cfg.span_years
    truth = {
        "scene": scene_index,
        "has_source": has_source,
        "source_x": sx, "source_y": sy, "depth_m": depth,
        "dv_m3": dv_m3, "peak_los_mm": peak,
        "grid_x": x, "grid_y": y,
        "los_unit": n_los,
        "n_epochs": T, "dt_days": cfg.dt_days,
        "span_years": cfg.span_years,
        "defo_final_mm": defo_final_mm,
        "rate_mm_per_yr": rate_true,
        "mean_coh_target": mean_coherence,
        "mean_coh_est": mean_gamma_est,
        "coverage": coverage,
        "atm_rms_mm": a_atm,
        "atm_rms_est_mm": atm_est,
        "goal": goal, "priority": priority,
    }
    return feature, truth
