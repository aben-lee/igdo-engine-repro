"""Detection-level metrics: recall, FAR, location error, alert latency."""
from __future__ import annotations

import math
from typing import Any, Dict

import numpy as np

LOC_TOL_M = 150.0


def score_window(result: Dict[str, Any], truth: Dict[str, Any],
                 loc_tol_m: float = LOC_TOL_M) -> Dict[str, Any]:
    """Compare runner output against the generator's hidden truth."""
    alerts = result.get("alerts", [])
    n_alerts = len(alerts)
    rec = {
        "has_event": bool(truth["has_event"]),
        "n_alerts": n_alerts,
        "detected_any": False,
        "correct_detect": False,
        "correct_detect_3d": False,
        "horiz_error_m": np.nan,
        "depth_error_m": np.nan,
        "min_loc_error_m": np.nan,
        "alert_latency_s": np.nan,
        "false_alarm": False,
        "n_stations": np.nan,
        "loc_rms_samples": np.nan,
        "composition_s": result.get("composition_s", np.nan),
        "node_cpu_s": result.get("node_cpu_s", np.nan),
        "proc_nominal_s": result.get("proc_nominal_s", np.nan),
        "rate_up": result.get("rate_up", False),
        "chunk_s": result.get("chunk_s", np.nan),
    }

    if truth["has_event"]:
        if n_alerts > 0:
            rec["detected_any"] = True
            errs, hidx, didx = [], [], []
            true_xyz = truth["event_xyz"]
            for k, a in enumerate(alerts):
                if a["xyz"] is not None:
                    d3 = float(np.linalg.norm(a["xyz"] - true_xyz))
                    dh = float(np.hypot(a["xyz"][0] - true_xyz[0],
                                        a["xyz"][1] - true_xyz[1]))
                    errs.append((dh, d3, k))
                    hidx.append(dh)
                    didx.append(abs(a["xyz"][2] - true_xyz[2]))
            if errs:
                # match the alert with the smallest horizontal error
                dh, d3, k = min(errs, key=lambda t: t[0])
                a = alerts[k]
                rec["horiz_error_m"] = dh
                rec["min_loc_error_m"] = d3
                rec["depth_error_m"] = abs(a["xyz"][2] - true_xyz[2])
                # recall uses epicentral accuracy (microseismic convention;
                # depth is intrinsically weaker for a limited-aperture array)
                rec["correct_detect"] = dh <= loc_tol_m
                rec["correct_detect_3d"] = d3 <= loc_tol_m
                fs = truth["fs"]
                chunk = result["chunk_s"]
                ev = truth["event_sample"]
                chunk_n = chunk * fs
                # alert fires when the chunk containing the final arrival closes
                chunk_close = math.ceil((a["alert_sample"] + 1) / chunk_n) * chunk_n
                wait = chunk_close / fs - ev / fs
                rec["alert_latency_s"] = wait + result["proc_nominal_s"]
                rec["n_stations"] = a["n_stations"]
                rec["loc_rms_samples"] = a["loc_rms_samples"]
    else:
        rec["false_alarm"] = n_alerts > 0
    return rec
