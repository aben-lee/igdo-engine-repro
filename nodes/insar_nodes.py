"""InSAR processing nodes for the cross-modality experiment.

The node set mirrors the microseismic paradigm one-for-one:

    static backbone : coh_mask -> per_pixel_rate -> threshold_detect -> alert
    IGDO conditional: gap_repair_insar / spatial_filter (adaptive front-end)
                      -> temporal_persist_check -> mogi_cross_validate
                      -> mogi_inversion (physics-based back-end)
                      investigation_rate_up (intent-driven scheduling)

Temporal persistence rejects single-epoch atmospheric bumps (not sustained);
the Mogi cross-validation rejects spatially diffuse/non-concentric residuals
that cannot be fit by a compact point source within the monitored area.

The Mogi unit response used by the inversion is the same analytic form as the
generator; successfully recovering the synthetic source cross-checks both.
"""
from __future__ import annotations

import math
import time
from typing import Any, Dict, List

import numpy as np
from scipy.ndimage import binary_opening, gaussian_filter, label

MONITOR_XY = 850.0   # m, inversion source must lie within the surveyed area


# --------------------------------------------------------------------------
# Forward response for inversion
# --------------------------------------------------------------------------
def mogi_los_rate_response(X, Y, sx, sy, depth, n_los, nu=0.25):
    """LOS rate (mm/yr) per unit volume-change rate (m^3/yr), Mogi model."""
    c_unit = (1.0 - nu) / math.pi
    dx = X - sx
    dy = Y - sy
    r3 = (dx ** 2 + dy ** 2 + depth ** 2) ** 1.5
    ux = c_unit * dx / r3
    uy = c_unit * dy / r3
    uz = c_unit * depth / r3
    los_m_per_m3yr = n_los[0] * ux + n_los[1] * uy + n_los[2] * uz
    return 1000.0 * los_m_per_m3yr


# --------------------------------------------------------------------------
# Backbone nodes
# --------------------------------------------------------------------------
def node_coh_mask(ctx: Dict[str, Any]) -> None:
    gate = ctx["params"].get("gate", 0.35)
    mean_coh = ctx["coh"].mean(axis=0)
    ctx["valid_mask"] = mean_coh >= gate
    ctx["mean_coh_map"] = mean_coh
    ctx["coverage_frac"] = float(ctx["valid_mask"].mean())


def node_per_pixel_rate(ctx: Dict[str, Any]) -> None:
    """Weighted/ordinary least-squares LOS rate (mm/yr) and linear R^2."""
    T = ctx["los"].shape[0]
    dt = ctx["dt_days"]
    t = np.arange(T) * dt / 365.25
    tc = t - t.mean()
    denom = float((tc ** 2).sum())
    valid = ctx["valid_mask"]
    los = ctx["los"]
    rate = np.full(los.shape[1:], np.nan)
    r2 = np.full(los.shape[1:], np.nan)
    idx = np.where(valid)
    lv = los[:, idx[0], idx[1]]                  # (T, P)
    slope = (tc[:, None] * lv).sum(axis=0) / denom
    intercept = lv.mean(axis=0) - slope * t.mean()
    pred = intercept[None, :] + slope[None, :] * t[:, None]
    ss_res = ((lv - pred) ** 2).sum(axis=0)
    ss_tot = ((lv - lv.mean(axis=0)[None, :]) ** 2).sum(axis=0) + 1e-12
    rate[idx] = slope
    r2[idx] = 1.0 - ss_res / ss_tot
    ctx["rate"] = rate
    ctx["rate_r2"] = r2


def _detect_clusters(rate, valid, rate_gate, min_cluster, x, y):
    thr = np.abs(np.where(valid, rate, np.nan))
    exceed = valid & (np.nan_to_num(thr, nan=0.0) >= rate_gate)
    lab, n = label(exceed, structure=np.ones((3, 3)))
    cands = []
    for cid in range(1, n + 1):
        m = lab == cid
        size = int(m.sum())
        if size < min_cluster:
            continue
        ys, xs = np.where(m)
        vals = rate[ys, xs]
        wfull = np.abs(vals)
        # Locate on the high-amplitude CORE (>= half the cluster peak): the
        # Mogi near-source core is compact and concentric, so its weighted
        # centroid is insensitive to edge clipping of the broad outer disc.
        core = wfull >= 0.5 * wfull.max()
        ys_c, xs_c, w = ys[core], xs[core], wfull[core]
        cx = float(np.sum(x[xs_c] * w) / w.sum())
        cy = float(np.sum(y[ys_c] * w) / w.sum())
        k = int(np.argmax(wfull))
        cands.append({
            "cluster_mask": m,
            "size": size,
            "peak_rate": float(vals[k]),
            "center_x": cx,
            "center_y": cy,
            "center_ij": (int(ys[k]), int(xs[k])),
        })
    cands.sort(key=lambda c: -abs(c["peak_rate"]))
    return cands


def node_threshold_detect(ctx: Dict[str, Any]) -> None:
    p = ctx["params"]
    ctx["candidates"] = _detect_clusters(
        ctx["rate"], ctx["valid_mask"], p.get("rate_gate_mm_yr", 12.0),
        p.get("min_cluster_pixels", 8), ctx["x"], ctx["y"])


def node_alert(ctx: Dict[str, Any]) -> None:
    det = ctx.get("detections")
    if det is None:
        # static path: every detected cluster is alerted without confirmation
        det = []
        for c in ctx.get("candidates", []):
            det.append({**c, "confirmed": False,
                        "loc_x": c["center_x"], "loc_y": c["center_y"]})
    ctx["alerts"] = det


# --------------------------------------------------------------------------
# Conditional adaptive/confirmation nodes
# --------------------------------------------------------------------------
def node_gap_repair_insar(ctx: Dict[str, Any]) -> None:
    """Tighten the mask: require coherence above gate in most epochs and
    remove speckle voids (neighbourhood opening)."""
    gate = ctx["params"].get("gate", 0.35)
    frac_req = ctx["params"].get("gap_epoch_frac", 0.6)
    valid_frac = (ctx["coh"] >= gate).mean(axis=0)
    mask = valid_frac >= frac_req
    mask = binary_opening(mask, structure=np.ones((3, 3)))
    ctx["valid_mask"] = mask
    ctx["coverage_frac"] = float(mask.mean())
    ctx["gap_repaired"] = True


def node_spatial_filter(ctx: Dict[str, Any]) -> None:
    """Edge-preserving adaptive smoothing of the rate field over valid pixels."""
    sigma = ctx["params"].get("spatial_sigma_px", 1.5)
    valid = ctx["valid_mask"].astype(float)
    r = np.nan_to_num(ctx["rate"], nan=0.0) * valid
    sr = gaussian_filter(r, sigma=sigma, mode="reflect") / (
        gaussian_filter(valid, sigma=sigma, mode="reflect") + 1e-12)
    ctx["rate"] = np.where(ctx["valid_mask"], sr, np.nan)
    ctx["spatially_filtered"] = True


def node_temporal_persist_check(ctx: Dict[str, Any]) -> None:
    """Require the anomaly to be sustained/linear over epochs (reject weather).
    Following SBAS multi-looking, the test fits a linear trend to the SPATIALLY
    AVERAGED LOS time series over each cluster's high-amplitude core pixels:
    the coherent crustal signal adds in phase while short-correlation phase
    noise decays as 1/sqrt(N), whereas a purely atmospheric patch leaves no
    significant temporal trend."""
    p = ctx["params"]
    r2_gate = p.get("persist_r2", 0.60)
    los = ctx["los"]
    T = los.shape[0]
    t = (np.arange(T) * ctx["dt_days"]) / 365.25
    tc = t - t.mean()
    denom_t = float(np.sum(tc * tc))
    rate = np.nan_to_num(ctx["rate"], nan=0.0)
    kept = []
    for c in ctx.get("candidates", []):
        m = c["cluster_mask"] & ctx["valid_mask"]
        core = m & (np.abs(rate) >= 0.6 * abs(c["peak_rate"]))
        n_core = int(core.sum())
        if n_core < 8:
            c["persist_frac"] = 0.0
            c["persist"] = False
            continue
        series = los[:, core].mean(axis=1)  # multi-looked core deformation
        slope = float(np.sum(tc * series) / denom_t)
        fit = slope * tc + series.mean()
        ss_res = float(np.sum((series - fit) ** 2))
        ss_tot = float(np.sum((series - series.mean()) ** 2))
        r2 = 1.0 - ss_res / max(ss_tot, 1e-12) if ss_tot > 1e-8 else 0.0
        c["persist_frac"] = float(r2)
        c["persist"] = bool(r2 >= r2_gate)
        if c["persist"]:
            kept.append(c)
    ctx["candidates"] = kept
    ctx["persisted"] = True


def _fit_grid(Xw, Yw, rw, n_los, sxg, syg, dg):
    """Vectorised brute-force Mogi search over paired horizontal grid points
    sxg/syg (shape Ns,) and depths dg (Nd,); data pixels Np.

    Note: no planar-ramp term is added. A ramp is strongly collinear with a
    deep/broad Mogi field and biases the recovered volume; long-wavelength
    atmosphere is instead rejected upstream by the temporal-persistence and
    Mogi morphology tests. The remaining location/volume spread at low
    coherence is a physical, reported limitation of single-track LOS
    inversion (depth-volume degeneracy), not a tuning residual.
    """
    n_up, nx_, ny_ = float(n_los[2]), float(n_los[0]), float(n_los[1])
    Xw = np.asarray(Xw)
    Yw = np.asarray(Yw)
    Np = Xw.size
    DX = Xw.reshape(1, 1, Np) - np.asarray(sxg).reshape(1, -1, 1)
    DY = Yw.reshape(1, 1, Np) - np.asarray(syg).reshape(1, -1, 1)
    DD = np.asarray(dg, dtype=float).reshape(-1, 1, 1)
    R3 = (DX * DX + DY * DY + DD * DD) ** 1.5
    K = (1.0 - 0.25) / math.pi * 1000.0
    G = K * (nx_ * DX / R3 + ny_ * DY / R3 + n_up * DD / R3)  # (Nd,Ns,Np)
    GtR = np.einsum("dsp,p->ds", G, rw)
    GtG = np.einsum("dsp,dsp->ds", G, G) + 1e-12
    dv = GtR / GtG
    rwrw = float(rw @ rw) + 1e-12
    nrss2 = (rwrw - 2.0 * dv * GtR + dv * dv * GtG) / rwrw
    nrss = np.sqrt(np.clip(nrss2, 0.0, None))
    nrss = np.where(dv > 0, nrss, np.inf)
    kd, ks = np.unravel_index(int(np.argmin(nrss)), nrss.shape)
    d = float(np.asarray(dg)[kd])
    dvk = float(dv[kd, ks])
    peak = dvk * K * n_up / (d * d)
    return {"sx": float(np.asarray(sxg)[ks]), "sy": float(np.asarray(syg)[ks]),
            "depth": d, "dv_rate": dvk, "nrss": float(nrss[kd, ks]),
            "pred_peak_rate": float(peak)}


def _mogi_fit_candidate(c, ctx, p):
    """Coarse-to-fine Mogi source inversion over the FULL valid scene. The
    coarse stage searches the whole monitoring area (independent of the
    possibly biased detection centroid, using both flanks of the anomaly),
    then the fine stage refines to pixel spacing."""
    X, Y, valid = ctx["X"], ctx["Y"], ctx["valid_mask"]
    n_los = ctx["n_los"]
    spacing = ctx["spacing_m"]
    dmin = p.get("depth_min_m", 300.0)
    dmax = p.get("depth_max_m", 1100.0)
    if valid.sum() < 30:
        return None
    Xw, Yw, rw = X[valid], Y[valid], ctx["rate"][valid]

    def clip_xy(v):
        return float(min(max(v, -MONITOR_XY), MONITOR_XY))

    gx = np.arange(-700.0, 700.0 + 1, 120.0)
    SX, SY = np.meshgrid(gx, gx)
    coarse = _fit_grid(Xw, Yw, rw, n_los, SX.ravel(), SY.ravel(),
                       np.arange(dmin, dmax + 1, 200.0))
    if coarse is None:
        return None
    fx = np.arange(clip_xy(coarse["sx"] - 120.0),
                   clip_xy(coarse["sx"] + 120.0) + 1, spacing)
    fy = np.arange(clip_xy(coarse["sy"] - 120.0),
                   clip_xy(coarse["sy"] + 120.0) + 1, spacing)
    FX, FY = np.meshgrid(fx, fy)
    fd = np.arange(max(dmin, coarse["depth"] - 200.0),
                   min(dmax, coarse["depth"] + 200.0) + 1, 100.0)
    fine = _fit_grid(Xw, Yw, rw, n_los, FX.ravel(), FY.ravel(), fd)
    return fine if fine["nrss"] <= coarse["nrss"] else coarse


def node_mogi_cross_validate(ctx: Dict[str, Any]) -> None:
    p = ctx["params"]
    res_gate = p.get("mogi_residual_gate", 0.55)
    rate_gate = p.get("rate_gate_mm_yr", 20.0)
    kept = []
    for c in ctx.get("candidates", []):
        fit = _mogi_fit_candidate(c, ctx, p)
        if fit is None:
            continue
        ok_res = fit["nrss"] <= res_gate
        ok_peak = abs(fit["pred_peak_rate"]) >= rate_gate
        ok_vol = (abs(fit["sx"]) <= MONITOR_XY and abs(fit["sy"]) <= MONITOR_XY)
        if ok_res and ok_peak and ok_vol:
            c["mogi"] = fit
            c["confirmed"] = True
            c["loc_x"] = fit["sx"]
            c["loc_y"] = fit["sy"]
            kept.append(c)
    ctx["candidates"] = kept
    ctx["cross_validated"] = True


def node_mogi_inversion(ctx: Dict[str, Any]) -> None:
    span = ctx["span_years"]
    for c in ctx.get("candidates", []):
        if "mogi" not in c:
            continue
        m = c["mogi"]
        c["inversion"] = {
            "x_m": m["sx"], "y_m": m["sy"], "depth_m": m["depth"],
            "dv_m3": m["dv_rate"] * span,
            "dv_rate_m3_per_yr": m["dv_rate"],
            "nrss": m["nrss"],
            "x_uncert_m": ctx["spacing_m"],
            "depth_uncert_m": ctx["params"].get("depth_step_m", 150.0),
        }
    ctx["detections"] = [c for c in ctx.get("candidates", []) if "mogi" in c]
    if ctx["detections"]:
        ctx["hazard_volume"] = True


def node_investigation_rate_up(ctx: Dict[str, Any]) -> None:
    """Switch to investigation mode: denser revisit, request high-res/ascent."""
    ctx["rate_up"] = True
    ctx["params"]["dt_days"] = ctx["dt_days"] / 3.0
    ctx["params"]["request_ascent_track"] = True


NODE_REGISTRY = {
    "coh_mask": node_coh_mask,
    "per_pixel_rate": node_per_pixel_rate,
    "threshold_detect": node_threshold_detect,
    "gap_repair_insar": node_gap_repair_insar,
    "spatial_filter": node_spatial_filter,
    "temporal_persist_check": node_temporal_persist_check,
    "mogi_cross_validate": node_mogi_cross_validate,
    "mogi_inversion": node_mogi_inversion,
    "investigation_rate_up": node_investigation_rate_up,
    "alert": node_alert,
}


def execute_node(node_id: str, ctx: Dict[str, Any]) -> float:
    """Execute one node, returning its measured wall-clock CPU time (s)."""
    t0 = time.perf_counter()
    NODE_REGISTRY[node_id](ctx)
    return time.perf_counter() - t0
