"""Synthetic microseismic waveform generator for the IGDO-Engine study.

Physical model (intentionally simple and fully specified in the paper):
  * Ricker wavelet source (fp ~ 80-150 Hz) at a known 3-D source location;
  * homogeneous velocity model, P-wave travel time = distance / vp;
  * geometric spreading amplitude decay 1/r;
  * AR(1) coloured background noise calibrated to a requested trace-gather
    mean SNR (signal/noise RMS, 20 log10);
  * bad channels: dead (flat) or noisy (extra noise), the counts setting
    ``quality.completeness``;
  * ingest-side quality estimated FROM THE DATA (noise-floor MAD vs window
    RMS, channel power checks) -- never copied from ground truth.

The generator returns (feature, truth): the GPJSON 4.1 Feature is the only
object visible to both pipelines; ``truth`` is used only for scoring.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, Optional, Tuple

import numpy as np

from gpjson41.builder import GPJSONBuilder
from gpjson41.policies_microseismic import quality_from_estimates


@dataclass
class MSConfig:
    fs: float = 1000.0           # Hz
    window_s: float = 5.0       # s
    n_stations: int = 8
    vp: float = 4500.0          # m/s, P-wave speed
    fp: float = 100.0           # Hz, Ricker central frequency
    noise_rho: float = 0.85     # AR(1) noise colour
    noise_seconds: float = 0.5  # noise-only leader used for quality estimation
    # survey geometry (local mine grid, metres)
    xlim: Tuple[float, float] = (-300.0, 300.0)
    ylim: Tuple[float, float] = (-150.0, 150.0)
    zlim: Tuple[float, float] = (-120.0, 40.0)
    base_amplitude: float = 1.0
    goal: str = "rockburst_event_detection"
    latency_budget_s: float = 4.0
    priority: str = "normal"
    near_boundary: bool = False
    crs: str = "local:mine_grid"
    inline_limit_bytes: int = 4 * 1024 * 1024  # keep gathers in-memory in the experiment
    # Transient mining interference (mechanical vibration / out-of-band
    # bursts / single-channel knocks). Expected count per window follows
    # _glitch_rate(snr); this is the dominant false-alarm source in reality.
    glitch_enabled: bool = True

    @property
    def n(self) -> int:
        return int(round(self.fs * self.window_s))


def glitch_rate(snr_db: float) -> float:
    """Expected number of transient interferers per window vs noise level."""
    return float(np.clip(2.2 * 10.0 ** (-snr_db / 18.0), 0.0, 3.0))


def _damped_sine(n_samp: int, fs: float, freq: float, tau: float,
                 phase: float) -> np.ndarray:
    t = np.arange(n_samp) / fs
    return np.exp(-t / tau) * np.sin(2.0 * math.pi * freq * t + phase)


def _inject_glitches(wf: np.ndarray, fs: float, scales: np.ndarray,
                     snr_db: float, rng: np.random.Generator,
                     leader_s: float) -> list:
    """Add realistic non-seismic transients; returns a truth record list."""
    rate = glitch_rate(snr_db)
    k_total = int(rng.poisson(rate))
    n_ch, n_samp = wf.shape
    records = []
    for _ in range(k_total):
        start = int(rng.integers(int(leader_s * fs), n_samp - int(0.4 * fs)))
        kind = "coherent_mechanical" if rng.random() < 0.7 else "impulse"
        if kind == "coherent_mechanical":
            # multi-channel, near-simultaneous (NOT a propagating point
            # source), low-frequency out-of-band damped vibration
            n_aff = int(rng.integers(4, n_ch + 1))
            chs = rng.choice(n_ch, size=n_aff, replace=False)
            freq = float(rng.uniform(15.0, 45.0))
            tau = float(rng.uniform(0.05, 0.12))
            dur = int(6 * tau * fs)
            for ch in chs:
                jit = int(rng.integers(-4, 5))
                s0 = max(0, start + jit)
                w = _damped_sine(dur, fs, freq, tau, rng.uniform(0, 2 * math.pi))
                amp = scales[ch] * float(rng.uniform(5.0, 10.0))
                end = min(n_samp, s0 + dur)
                wf[ch, s0:end] += amp * w[: end - s0]
            records.append({"kind": kind, "sample": start, "freq": freq,
                            "channels": chs.tolist()})
        else:
            # 1-2 channel broad-band knock: cannot associate on its own
            n_aff = int(rng.integers(1, 3))
            chs = rng.choice(n_ch, size=n_aff, replace=False)
            freq = float(rng.uniform(60.0, 150.0))
            tau = float(rng.uniform(0.01, 0.03))
            dur = int(6 * tau * fs)
            for ch in chs:
                w = _damped_sine(dur, fs, freq, tau, rng.uniform(0, 2 * math.pi))
                amp = scales[ch] * float(rng.uniform(6.0, 12.0))
                end = min(n_samp, start + dur)
                wf[ch, start:end] += amp * w[: end - start]
            records.append({"kind": kind, "sample": start, "freq": freq,
                            "channels": chs.tolist()})
    return records


def ricker_wavelet(t: np.ndarray, fp: float) -> np.ndarray:
    a = (math.pi * fp * t) ** 2
    return (1.0 - 2.0 * a) * np.exp(-a)


def make_station_network(cfg: MSConfig, seed: int = 0) -> np.ndarray:
    """Deterministic 3-D deep-mine array (8 stations).

    Six rib/roof/floor stations line two roadways; two borehole stations at
    depth give vertical aperture, without which arrival-time differences
    cannot resolve hypocentre depth (the classic planar-array ambiguity).
    """
    rng = np.random.default_rng(seed)
    xs = np.linspace(cfg.xlim[0] * 0.7, cfg.xlim[1] * 0.7, 3)
    ys = np.array([cfg.ylim[0] * 0.6, cfg.ylim[1] * 0.6])
    grid = []
    for y in ys:
        for xi, x in enumerate(xs):
            z = 10.0 if xi % 2 == 0 else -30.0   # roof / floor alternating
            grid.append([x, y, z])
    # two down-hole geophones for vertical constraint
    grid.append([-120.0, 0.0, -80.0])
    grid.append([120.0, 0.0, -140.0])
    grid = np.asarray(grid, dtype=np.float64)
    grid += rng.normal(scale=6.0, size=grid.shape)
    return grid


def reference_signal_rms(cfg: MSConfig, stations: np.ndarray) -> float:
    """Median channel RMS of a reference event at the volume centre.

    Used to define an absolute noise-floor quality scale ('expected SNR of a
    typical event under the current noise'), which is well-defined even in
    event-free windows.
    """
    n_samp = cfg.n
    fs = cfg.fs
    centre = np.array([np.mean(cfg.xlim), np.mean(cfg.ylim), np.mean(cfg.zlim)])
    t = np.arange(n_samp) / fs
    ref = np.zeros((len(stations), n_samp))
    for i, s in enumerate(stations):
        r = max(20.0, float(np.linalg.norm(s - centre)))
        tt = r / cfg.vp
        ref[i] = (cfg.base_amplitude / r * 120.0) * ricker_wavelet(
            t - tt - 0.5 * cfg.window_s, cfg.fp)
    return float(np.median(np.sqrt((ref ** 2).mean(axis=1))))


def _colored_noise(n_samp: int, n_ch: int, rho: float, rng: np.random.Generator) -> np.ndarray:
    """Stationary AR(1) noise; variance normalised to 1 per channel."""
    w = rng.standard_normal(size=(n_ch, n_samp))
    out = np.zeros_like(w)
    out[:, 0] = w[:, 0]
    for k in range(1, n_samp):
        out[:, k] = rho * out[:, k - 1] + math.sqrt(1.0 - rho ** 2) * w[:, k]
    out /= out.std(axis=1, keepdims=True) + 1e-12
    return out


def _estimate_ingest_quality(
    wf: np.ndarray, cfg: MSConfig, ref_sig_rms: float
) -> Tuple[float, np.ndarray, np.ndarray]:
    """Return (expected_snr_db, flagged-dead mask, flagged-noisy mask).

    Quality is derived from the noise-only leader: expected SNR of a reference
    event under the current noise floor -- well defined for event-free windows.
    """
    n0 = int(cfg.fs * cfg.noise_seconds)
    lead = wf[:, :n0]
    noise_rms = np.sqrt((lead ** 2).mean(axis=1))
    # bad-channel tests
    med_noise = np.median(noise_rms[noise_rms > 1e-9]) if np.any(noise_rms > 1e-9) else 1.0
    dead = (lead.std(axis=1) < 1e-3 * (med_noise + 1e-12)) | (np.abs(wf).max(axis=1) < 5e-3)
    noisy = noise_rms > 2.5 * med_noise
    good = ~(dead | noisy)
    floor = float(np.median(noise_rms[good])) if good.sum() >= 3 else float(med_noise)
    est_snr = 20.0 * math.log10((ref_sig_rms + 1e-12) / (floor + 1e-12))
    return est_snr, dead, noisy


def generate_window(
    cfg: MSConfig,
    stations: np.ndarray,
    snr_db: float,
    completeness: float,
    has_event: bool,
    rng: np.random.Generator,
    ref_sig_rms: float,
    window_index: int = 0,
    near_boundary: Optional[bool] = None,
    priority: Optional[str] = None,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Generate one processing window as a GPJSON 4.1 seismic.trace Feature."""
    n_ch, n_samp = len(stations), cfg.n
    fs = cfg.fs
    near_boundary = cfg.near_boundary if near_boundary is None else near_boundary
    priority = cfg.priority if priority is None else priority

    # --- source ----------------------------------------------------------
    if has_event:
        if near_boundary:
            # within ~40 m of a roadway rib
            ex = float(rng.uniform(*cfg.xlim))
            ey = float(cfg.ylim[1] * 0.6 + rng.uniform(-30, 30))
            ez = float(rng.uniform(*cfg.zlim))
        else:
            ex = float(rng.uniform(*cfg.xlim))
            ey = float(rng.uniform(*cfg.ylim))
            ez = float(rng.uniform(*cfg.zlim))
        event_xyz = np.array([ex, ey, ez])
        event_sample = int(rng.integers(int(0.18 * n_samp), int(0.82 * n_samp)))
    else:
        event_xyz = None
        event_sample = None

    # --- clean signal ----------------------------------------------------
    sig = np.zeros((n_ch, n_samp), dtype=np.float64)
    if has_event:
        t = np.arange(n_samp) / fs
        for i, s in enumerate(stations):
            r = max(20.0, float(np.linalg.norm(s - event_xyz)))
            tt = r / cfg.vp
            amp = cfg.base_amplitude / r * 120.0  # geometric spreading normalised
            sig[i] = amp * ricker_wavelet(t - tt - event_sample / fs, cfg.fp)

    # --- coloured noise, calibrated to the requested noise floor --------
    # The noise level is set so that a REFERENCE event at the volume centre
    # would meet the target SNR. This keeps the noise floor identical across
    # event / event-free windows and makes SNR meaningful in both.
    noise = _colored_noise(n_samp, n_ch, cfg.noise_rho, rng)
    noise_rms_unit = np.sqrt((noise ** 2).mean(axis=1))
    target_ratio = 10.0 ** (snr_db / 20.0)
    scales = ref_sig_rms / (noise_rms_unit * target_ratio + 1e-12)
    scales = scales * rng.uniform(0.7, 1.4, size=n_ch)
    wf = sig + noise * scales[:, None]

    # --- transient mining interference (dominant false-alarm source) ----
    glitches = (_inject_glitches(wf, fs, scales, snr_db, rng,
                                 cfg.noise_seconds)
                if cfg.glitch_enabled else [])

    # --- bad channels ----------------------------------------------------
    n_bad = int(round((1.0 - completeness) * n_ch))
    all_idx = np.arange(n_ch)
    bad_idx = rng.choice(all_idx, size=n_bad, replace=False) if n_bad > 0 else np.array([], int)
    dead_true, noisy_true = [], []
    for k, ch in enumerate(bad_idx):
        if k % 2 == 0:  # dead / flat channel
            wf[ch, :] = rng.normal(scale=1e-4, size=n_samp)
            dead_true.append(ch)
        else:           # noisy channel: extra loud background
            wf[ch, :] = sig[ch] + noise[ch] * scales[ch] * rng.uniform(5.0, 10.0)
            noisy_true.append(ch)
    dead_true = np.array(dead_true, int)
    noisy_true = np.array(noisy_true, int)

    # --- ingest estimates quality FROM THE DATA --------------------------
    est_snr, dead_flag, noisy_flag = _estimate_ingest_quality(wf, cfg, ref_sig_rms)
    n_flagged = int((dead_flag | noisy_flag).sum())
    n_good = n_ch - n_flagged
    qmap = quality_from_estimates(est_snr, n_good, n_ch, n_flagged)

    # --- assemble GPJSON 4.1 Feature ------------------------------------
    t0 = datetime(2026, 4, 21, 0, 0, 0, tzinfo=timezone.utc) + timedelta(
        seconds=window_index * cfg.window_s)
    t1 = t0 + timedelta(seconds=cfg.window_s)
    fid = f"ms_gather_w{window_index:04d}"
    center = stations.mean(axis=0)

    b = GPJSONBuilder("seismic.trace", fid, crs=cfg.crs)
    b.set_point3d(center)
    b.set_source("synth_ingest_node@repro")
    b.set_timestamp(t0.strftime("%Y-%m-%dT%H:%M:%SZ"))
    b.add_quality(
        completeness=qmap["completeness"],
        outlier_rate=qmap["outlierRate"],
        confidence=qmap["confidence"],
        validity=qmap["validity"],
        uncertainty={"type": "gaussian", "sigma": float(qmap["estSnrDb"]),
                     "note": "estSnrDb stored in health"},
    )
    b.set_pipeline(
        event_time=t0.strftime("%Y-%m-%dT%H:%M:%SZ"),
        watermark=t1.strftime("%Y-%m-%dT%H:%M:%SZ"),
        latency_budget=cfg.latency_budget_s,
        qos={"priority": priority, "deliveryGuarantee": "at_least_once"},
        stage={"level": "raw", "readyForFusion": False},
        health={
            "sensorStatus": "degraded" if n_flagged else "OK",
            "estSnrDb": qmap["estSnrDb"],
            "deadChannels": dead_flag.tolist(),
            "noisyChannels": noisy_flag.tolist(),
        },
        transport={"protocol": "mqtt", "topic": "mine/mseis/gather"},
    )
    b.set_intent(
        goal=cfg.goal + ("_near_boundary" if near_boundary else ""),
        outcomes=["event_catalog", "alert_event"],
        constraints={"latencyBudget_s": cfg.latency_budget_s,
                     "minStations": 3},
    )
    b.add_property("fs", cfg.fs)
    b.add_property("vp", cfg.vp)
    b.add_property("fp", cfg.fp)
    b.add_dataset_ndarray(
        "waveform", wf.astype(np.float32),
        axes=["channel", "sample"], units="counts",
        inline_limit_bytes=cfg.inline_limit_bytes,
    )
    b.add_dataset_sheet(
        "station_geometry",
        ["stationId", "x/m", "y/m", "z/m"],
        [[f"CH{i:02d}", float(s[0]), float(s[1]), float(s[2])]
         for i, s in enumerate(stations)],
    )
    feature = b.build()

    truth = {
        "window": window_index,
        "has_event": has_event,
        "event_xyz": event_xyz,
        "event_sample": event_sample,
        "event_time_s": (event_sample / fs) if has_event else None,
        "stations": stations,
        "fs": fs,
        "vp": cfg.vp,
        "true_snr_db": snr_db,
        "completeness_target": completeness,
        "dead_channels": dead_true,
        "noisy_channels": noisy_true,
        "glitches": glitches,
        "est_snr_db": est_snr,
        "near_boundary": near_boundary,
        "priority": priority,
    }
    return feature, truth
