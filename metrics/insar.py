"""InSAR detection / inversion / routing metrics for the Mogi experiment.

Detection recall uses horizontal source-location accuracy at several
tolerances (the product carries 30 m pixels, so 60/90/120 m = 2/3/4 pixels);
depth and volume errors are reported separately because single-track LOS Mogi
inversion has an intrinsic depth-volume degeneracy. False-alarm rate is
measured on source-free scenes. Routing accuracy compares the data-driven
pipeline against the domain reference policy evaluated on GROUND-TRUTH
coherence/coverage.
"""
from __future__ import annotations

from typing import Any, Dict, List

import numpy as np

from gpjson41 import policies_insar as pol
from nodes.insar_nodes import mogi_los_rate_response

ACTION_UNIVERSE = pol.ACTION_UNIVERSE
LOC_TOLS = (60.0, 90.0, 120.0)
INVERSION_XY = 700.0


def _rate_field_nrmse(a: Dict[str, Any], truth: Dict[str, Any]) -> float:
    """Normalised RMSE of the inverted Mogi rate field vs. hidden truth."""
    if a.get("depth_m") is None or a.get("dv_m3") is None:
        return np.nan
    gx, gy = np.meshgrid(truth["grid_x"], truth["grid_y"])
    dv_rate = a["dv_m3"] / truth["span_years"]
    pred = mogi_los_rate_response(gx, gy, a["loc_x"], a["loc_y"],
                                  a["depth_m"], truth["los_unit"]) * dv_rate
    true_rate = truth["rate_mm_per_yr"]
    m = (np.abs(gx) <= INVERSION_XY) & (np.abs(gy) <= INVERSION_XY)
    rmse = float(np.sqrt(np.mean((pred[m] - true_rate[m]) ** 2)))
    rms = float(np.sqrt(np.mean(true_rate[m] ** 2))) + 1e-12
    return rmse / rms


def score_scene(result: Dict[str, Any], truth: Dict[str, Any],
                loc_tols=LOC_TOLS) -> Dict[str, Any]:
    alerts = result.get("alerts", [])
    true_peak_rate = (truth["peak_los_mm"] / truth["span_years"]
                      if truth["has_source"] else np.nan)
    rec = {
        "has_source": bool(truth["has_source"]),
        "mean_coh_target": truth["mean_coh_target"],
        "n_alerts": len(alerts),
        "detected_any": False,
        "confirmed": False,
        "horiz_error_m": np.nan,
        "depth_error_m": np.nan,
        "dv_rel_error_pct": np.nan,
        "peak_rate_err_pct": np.nan,
        "rate_field_nrmse": np.nan,
        "nrss": np.nan,
        "false_alarm": False,
        "coverage_frac": result.get("coverage_frac", np.nan),
        "composition_s": result.get("composition_s", np.nan),
        "node_cpu_s": result.get("node_cpu_s", np.nan),
        "proc_nominal_s": result.get("proc_nominal_s", np.nan),
        "rate_up": result.get("rate_up", False),
        "spatially_filtered": result.get("spatially_filtered", False),
        "gap_repaired": result.get("gap_repaired", False),
        "cross_validated": result.get("cross_validated", False),
    }
    for tol in loc_tols:
        rec[f"correct_{int(tol)}m"] = False

    if truth["has_source"]:
        if alerts:
            rec["detected_any"] = True
            a = min(alerts,
                    key=lambda z: np.hypot(z["loc_x"] - truth["source_x"],
                                           z["loc_y"] - truth["source_y"]))
            he = float(np.hypot(a["loc_x"] - truth["source_x"],
                                a["loc_y"] - truth["source_y"]))
            rec["horiz_error_m"] = he
            rec["confirmed"] = bool(a.get("confirmed"))
            rec["peak_rate_err_pct"] = (
                abs(a["peak_rate"] - true_peak_rate) / true_peak_rate * 100.0)
            for tol in loc_tols:
                rec[f"correct_{int(tol)}m"] = bool(he <= tol)
            if a.get("depth_m") is not None:
                rec["depth_error_m"] = abs(a["depth_m"] - truth["depth_m"])
                rec["nrss"] = a.get("nrss")
            if a.get("dv_m3") is not None:
                rec["dv_rel_error_pct"] = (
                    abs(a["dv_m3"] - truth["dv_m3"]) / truth["dv_m3"] * 100.0)
                rec["rate_field_nrmse"] = _rate_field_nrmse(a, truth)
    else:
        rec["false_alarm"] = len(alerts) > 0
    return rec


def expected_actions(truth: Dict[str, Any]) -> List[str]:
    """Reference policy actions under GROUND-TRUTH coherence/coverage."""
    q = pol.quality_from_coherence(truth["mean_coh_est"], truth["coverage"])
    state = {"quality": q, "health": {"meanCoherence": truth["mean_coh_est"]},
             "stage": "raw"}
    intent = {"goal": truth["goal"], "outcomes": [],
              "constraints": {"qos": {"priority": truth.get("priority",
                                                            "normal")}}}
    return pol.reference_actions(state, intent)


def routing_score(actual_actions: List[str], truth: Dict[str, Any]) -> Dict[str, Any]:
    exp = set(expected_actions(truth))
    act = set(actual_actions)
    correct = sum(1 for a in ACTION_UNIVERSE if (a in act) == (a in exp))
    return {
        "routing_exact": bool(exp == act),
        "routing_hamming": correct / len(ACTION_UNIVERSE),
        "expected_actions": ";".join(sorted(exp)),
        "actual_actions": ";".join(sorted(act)),
    }
