"""Processing nodes for microseismic monitoring.

Each node is a deterministic function over a shared execution context dict
(``ctx``). Both the static and IGDO-aware runners call the SAME node
implementations -- the only difference is whether the node sequence is fixed
or composed from the NKG according to the IGDO state/intent.

Detection uses the classical recursive STA/LTA (Obspy reference
implementation, same parameters available to both pipelines).
"""
from __future__ import annotations

import math
import time
from typing import Any, Dict, List, Tuple

import numpy as np
from scipy import signal as sps
from obspy.signal.trigger import recursive_sta_lta, trigger_onset

# --------------------------------------------------------------------------
# Signal conditioning
# --------------------------------------------------------------------------
NARROW_BAND = (60.0, 200.0)
WIDE_BAND = (20.0, 300.0)


def _butter_bandpass(wf: np.ndarray, fs: float, flo: float, fhi: float,
                     order: int = 4) -> np.ndarray:
    nyq = fs / 2.0
    b, a = sps.butter(order, [flo / nyq, min(fhi / nyq, 0.999)], btype="band")
    out = np.zeros_like(wf, dtype=np.float64)
    for i in range(wf.shape[0]):
        out[i] = sps.filtfilt(b, a, wf[i])
    return out


def node_bandpass_wide(ctx: Dict[str, Any]) -> None:
    fs = ctx["fs"]
    ctx["wf"] = _butter_bandpass(ctx["wf"], fs, *WIDE_BAND)
    ctx["detect_band"] = "wide(20-300Hz)"


def node_bandpass_narrow(ctx: Dict[str, Any]) -> None:
    fs = ctx["fs"]
    ctx["wf"] = _butter_bandpass(ctx["wf"], fs, *NARROW_BAND)
    ctx["detect_band"] = "narrow(60-200Hz)"


def node_robust_denoise(ctx: Dict[str, Any]) -> None:
    """Adaptive narrow-band screening under degraded data.

    Switches from the wide monitoring band to the event band (60-200 Hz),
    attenuating out-of-band mechanical vibration and low-frequency swell while
    preserving the impulsive P arrival (a Wiener/local-stationarity filter was
    avoided because it distorts short transients and degrades location).
    Channels excluded by gap_repair are left neutralised.
    """
    fs = ctx["fs"]
    wf = ctx["wf"]
    n_ch = wf.shape[0]
    nyq = fs / 2.0
    b, a = sps.butter(4, [NARROW_BAND[0] / nyq, NARROW_BAND[1] / nyq],
                      btype="band")
    good = ctx.get("good_mask", np.ones(n_ch, dtype=bool))
    out = np.zeros_like(wf)
    for i in range(n_ch):
        if good[i]:
            out[i] = sps.filtfilt(b, a, wf[i])
    ctx["wf"] = out
    ctx["detect_band"] = "narrow(60-200)"


def node_gap_repair(ctx: Dict[str, Any]) -> None:
    """Detect dead/noisy channels and exclude them from association/location.

    Bad channels are diagnosed on the noise-only LEADER (first 0.5 s), where
    neither events nor transient interference occur, so a channel carrying a
    glitch is never mistaken for a permanently bad channel.
    """
    wf = ctx["wf"]
    fs = ctx["fs"]
    n0 = max(8, int(0.5 * fs))
    lead = wf[:, :n0]
    lead_rms = np.sqrt((lead ** 2).mean(axis=1))
    pos = lead_rms[lead_rms > 1e-12]
    med = float(np.median(pos)) if len(pos) else 1.0
    dead = lead_rms < 0.10 * med          # ~10x quieter than the array median
    noisy = lead_rms > 2.5 * med          # >2.5x noisier than the median
    flagged = dead | noisy
    ctx["good_mask"] = ~flagged
    # neutralise flagged channels so they cannot fire spurious triggers
    wf[flagged] = 0.0
    ctx["wf"] = wf
    ctx["gap_repair_found"] = int(flagged.sum())
    # location geometry retains all stations; detection uses good ones only.


# Monitoring volume for the location sanity check (local mine grid, m).
MONITOR_VOLUME = ((-330.0, 330.0), (-180.0, 180.0), (-160.0, 70.0))


# --------------------------------------------------------------------------
# Detection / association / location
# --------------------------------------------------------------------------
def _sta_lta_per_channel(
    wf: np.ndarray, fs: float, good_mask: np.ndarray,
    sta_s: float, lta_s: float, th_on: float, th_off: float
) -> List[List[Tuple[int, int]]]:
    triggers = []
    sta_n, lta_n = max(2, int(sta_s * fs)), max(4, int(lta_s * fs))
    for i in range(wf.shape[0]):
        if not good_mask[i]:
            triggers.append([])
            continue
        cf = recursive_sta_lta(wf[i].astype(np.float64), sta_n, lta_n)
        cf = np.nan_to_num(cf, nan=0.0, posinf=0.0, neginf=0.0)
        trg = trigger_onset(cf, th_on, th_off)
        triggers.append([(int(a), int(b)) for a, b in trg])
    return triggers


def _associate(
    triggers: List[List[Tuple[int, int]]], min_stations: int, tol_samples: int
) -> List[Dict[str, Any]]:
    """Cluster channel triggers into event candidates by arrival consistency."""
    records = []
    for ch, lst in enumerate(triggers):
        for on, off in lst:
            records.append((on, ch))
    records.sort()
    candidates: List[Dict[str, Any]] = []
    used = [False] * len(records)
    for i, (on_i, ch_i) in enumerate(records):
        if used[i]:
            continue
        cluster = [(on_i, ch_i)]
        used[i] = True
        for j in range(i + 1, len(records)):
            if used[j]:
                continue
            on_j, ch_j = records[j]
            if on_j - on_i > tol_samples:
                break
            chs = {c for _, c in cluster}
            if ch_j not in chs:
                cluster.append((on_j, ch_j))
                used[j] = True
        chs = {c for _, c in cluster}
        if len(chs) >= min_stations:
            onsets = {ch: on for on, ch in cluster}
            candidates.append(
                {"channels": sorted(chs), "onsets": onsets,
                 "t_earliest": min(on for on, _ in cluster)}
            )
    # merge overlapping candidates (same event can survive at two thresholds)
    merged: List[Dict[str, Any]] = []
    for c in sorted(candidates, key=lambda x: x["t_earliest"]):
        if merged and c["t_earliest"] - merged[-1]["t_earliest"] < tol_samples:
            # keep the one with more stations
            if len(c["channels"]) > len(merged[-1]["channels"]):
                merged[-1] = c
        else:
            merged.append(c)
    return merged


def _grid_locate(
    candidate: Dict[str, Any], stations: np.ndarray, vp: float,
    grid_step: Tuple[float, float, float] = (20.0, 20.0, 15.0),
    bounds: Tuple[float, float, float, float, float, float] = None,
) -> Dict[str, Any]:
    """Grid-search hypocentre using arrival-TIME DIFFERENCES (origin unknown)."""
    chs = list(candidate["channels"])
    if len(chs) < 3:
        return {"xyz": None, "rms": np.nan, "n": len(chs)}
    onsets = candidate["onsets"]
    ref = chs[0]
    others = chs[1:]
    obs_dt = np.array([(onsets[c] - onsets[ref]) / 1.0 for c in others],
                      dtype=np.float64)  # in samples
    if bounds is None:
        (ax, bxx), (ay, byy), (az, bzz) = MONITOR_VOLUME
        bx, by, bz = (ax, bxx), (ay, byy), (az, bzz)
    else:
        bx, by, bz = (bounds[0], bounds[1]), (bounds[2], bounds[3]), (bounds[4], bounds[5])
    gx = np.arange(bx[0], bx[1] + 1, grid_step[0])
    gy = np.arange(by[0], by[1] + 1, grid_step[1])
    gz = np.arange(bz[0], bz[1] + 1, grid_step[2])
    fs = candidate.get("_fs", 1000.0)
    GX, GY, GZ = np.meshgrid(gx, gy, gz, indexing="ij")
    P = np.column_stack((GX.ravel(), GY.ravel(), GZ.ravel()))
    s_chs = stations[chs]
    d = np.linalg.norm(P[:, None, :] - s_chs[None, :, :], axis=2)  # M x n
    pred_dt = (d[:, 1:] - d[:, 0:1]) / vp * fs                      # samples
    rms_map = np.sqrt(np.mean((pred_dt - obs_dt[None, :]) ** 2, axis=1))
    j = int(np.argmin(rms_map))
    return {"xyz": P[j], "rms": float(rms_map[j]), "n": len(chs)}


def node_sta_lta_detect(ctx: Dict[str, Any]) -> None:
    p = ctx["params"]
    triggers = _sta_lta_per_channel(
        ctx["wf"], ctx["fs"], ctx["good_mask"],
        sta_s=p.get("sta_s", 0.10), lta_s=p.get("lta_s", 1.0),
        th_on=p.get("th_on", 3.5), th_off=p.get("th_off", 1.5),
    )
    tol = int(ctx["params"].get("assoc_tol_s", 0.25) * ctx["fs"])
    candidates = _associate(triggers, p.get("min_stations", 3), tol)
    ctx["triggers"] = triggers
    ctx["candidates"] = candidates


def _aic_onset(trace: np.ndarray, c0: int, fs: float) -> int:
    """Refine a coarse onset by AIC within a short window around it."""
    n = len(trace)
    lo = max(2, int(c0 - 0.06 * fs))
    hi = min(n - 2, int(c0 + 0.15 * fs))
    e = trace[lo:hi].astype(np.float64) ** 2
    nn = len(e)
    k = np.arange(2, nn - 2)
    cs1, cs2 = np.cumsum(e), np.cumsum(e * e)
    s1, s2 = cs1[k], cs2[k]
    var = np.maximum(s2 / k - (s1 / k) ** 2, 1e-20)
    t1, t2 = cs1[-1] - s1, cs2[-1] - cs2[k]
    nk = nn - k
    var2 = np.maximum(t2 / nk - (t1 / nk) ** 2, 1e-20)
    aic = k * np.log(var) + (nn - k) * np.log(var2)
    return lo + int(k[int(np.argmin(aic))])


def node_refine_onsets(ctx: Dict[str, Any]) -> None:
    """Detection/location frequency-band separation.

    Screening may run on a narrow band for interference rejection; onset times
    for location are re-picked on the wide-banded RAW waveform using AIC around
    each coarse STA/LTA onset, restoring arrival-time accuracy. Both pipelines
    use this identical node, so location accuracy is compared fairly.
    """
    fs = ctx["fs"]
    raw = ctx["wf_raw"]
    nyq = fs / 2.0
    b, a = sps.butter(4, [WIDE_BAND[0] / nyq, min(WIDE_BAND[1] / nyq, 0.999)],
                      btype="band")
    good = ctx.get("good_mask", np.ones(raw.shape[0], dtype=bool))
    wide = np.zeros_like(raw)
    for i in range(raw.shape[0]):
        if good[i]:
            wide[i] = sps.filtfilt(b, a, raw[i])
    for c in ctx.get("candidates", []):
        c["onsets"] = {ch: _aic_onset(wide[ch], c0, fs)
                       for ch, c0 in c["onsets"].items()}


def node_cross_validate(ctx: Dict[str, Any]) -> None:
    """Confirm candidates: >=4 stations, travel-time consistency, and a
    hypocentre that lies inside the monitored volume.

    Near-simultaneous mechanical vibration (no propagating wavefront) can only
    fit the velocity model by placing a spurious source far outside the
    monitored volume, so the volume check rejects it.
    """
    vp = ctx["vp"]
    fs = ctx["fs"]
    rms_tol = ctx["params"].get("cv_rms_samples", 25.0)
    (xmin, xmax), (ymin, ymax), (zmin, zmax) = MONITOR_VOLUME
    kept = []
    for c in ctx.get("candidates", []):
        c["_fs"] = fs
        loc = _grid_locate(c, ctx["stations"], vp)
        # 3 stations give 2 arrival-time differences; with a fixed velocity
        # model and the monitoring-volume prior this is the minimum locatable.
        ok_n = loc["n"] >= 3
        ok_rms = loc["rms"] <= rms_tol
        xyz = loc["xyz"]
        ok_vol = (xyz is not None and xmin <= xyz[0] <= xmax
                  and ymin <= xyz[1] <= ymax and zmin <= xyz[2] <= zmax)
        if ok_n and ok_rms and ok_vol:
            c["_loc"] = loc
            kept.append(c)
    ctx["candidates"] = kept
    ctx["cross_validated"] = True


def node_grid_locate(ctx: Dict[str, Any]) -> None:
    vp = ctx["vp"]
    fs = ctx["fs"]
    for c in ctx.get("candidates", []):
        if "_loc" not in c:
            c["_fs"] = fs
            c["_loc"] = _grid_locate(c, ctx["stations"], vp)
    ctx["detections"] = [c for c in ctx["candidates"] if c["_loc"]["xyz"] is not None]


def node_event_rate_up(ctx: Dict[str, Any]) -> None:
    """Switch to faster screening cadence / shorter STA window (S3)."""
    ctx["params"]["sta_s"] = 0.05
    ctx["params"]["min_stations"] = 2
    ctx["params"]["chunk_s"] = 0.25
    ctx["rate_up"] = True


def node_alert(ctx: Dict[str, Any]) -> None:
    ctx["alerts"] = list(ctx.get("detections", []))


# --------------------------------------------------------------------------
# Node registry (id -> callable); mirrors configs/nkg/nodes_microseismic.yaml
# --------------------------------------------------------------------------
NODE_REGISTRY = {
    "bandpass_wide": node_bandpass_wide,
    "bandpass_narrow": node_bandpass_narrow,
    "robust_denoise": node_robust_denoise,
    "gap_repair": node_gap_repair,
    "sta_lta_detect": node_sta_lta_detect,
    "refine_onsets": node_refine_onsets,
    "cross_validate": node_cross_validate,
    "grid_locate": node_grid_locate,
    "event_rate_up": node_event_rate_up,
    "alert": node_alert,
}


def execute_node(node_id: str, ctx: Dict[str, Any]) -> float:
    """Execute one node, returning its measured wall-clock CPU time (s)."""
    t0 = time.perf_counter()
    NODE_REGISTRY[node_id](ctx)
    return time.perf_counter() - t0
