"""InSAR domain adapter and run_static/run_igdo entry points.

Loads a Grid2D stack (LOS displacement + coherence cubes) rather than a
waveform gather; the execution core and planner are shared unchanged with the
microseismic adapter, evidencing the "one orchestration engine, modality
specific loaders/NKGs" architecture.
"""
from __future__ import annotations

from typing import Any, Dict, List, Tuple

import numpy as np

from gpjson41.parser import GPJSONParser
from gpjson41 import policies_insar
from nodes.insar_nodes import execute_node
from .cost_model import plan_processing_s
from .core import run_igdo_core, run_static_core

CONDITIONAL_ACTIONS = set(policies_insar.ACTION_UNIVERSE)


def _load_context(feature: Dict[str, Any]):
    p = GPJSONParser(feature)
    los = p.load_role_array("los_displacement").astype(np.float64)
    coh = p.load_role_array("coherence").astype(np.float64)
    n_los = np.asarray(p.load_role_array("los_unit_vector"), dtype=np.float64)
    geom = feature["geometry"]
    ox, oy = geom["origin"]
    dx, dy = geom["spacing"]
    ny, nx = geom["shape"]
    x = ox + np.arange(nx) * dx
    y = oy + np.arange(ny) * dy
    X, Y = np.meshgrid(x, y)
    dt_days = float(feature["properties"]["dt_days"])
    n_epochs = int(feature["properties"]["n_epochs"])
    span_years = (n_epochs - 1) * dt_days / 365.25
    ctx = {
        "los": los, "coh": coh, "n_los": n_los,
        "x": x, "y": y, "X": X, "Y": Y,
        "spacing_m": float(dx),
        "dt_days": dt_days, "span_years": span_years,
        "valid_mask": None, "rate": None, "rate_r2": None,
        "mean_coh_map": None, "coverage_frac": None,
        "params": {}, "state": None, "intent": None,
        "candidates": [], "detections": None, "alerts": [],
        "rate_up": False,
    }
    size_info = (ny * nx, n_epochs)
    return ctx, p, size_info


def _alert_record(c: Dict[str, Any]) -> Dict[str, Any]:
    rec = {
        "confirmed": bool(c.get("confirmed", False)),
        "loc_x": float(c.get("loc_x", c["center_x"])),
        "loc_y": float(c.get("loc_y", c["center_y"])),
        "peak_rate": float(c.get("peak_rate", 0.0)),
        "size": int(c.get("size", 0)),
        "depth_m": None, "dv_m3": None, "nrss": None,
    }
    if "mogi" in c:
        rec["depth_m"] = float(c["mogi"]["depth"])
        rec["nrss"] = float(c["mogi"]["nrss"])
    if c.get("inversion"):
        rec["dv_m3"] = float(c["inversion"]["dv_m3"])
    return rec


def _summarise(ctx: Dict[str, Any], plan: List[str], size_info: Tuple[int, int],
               nkg, cpu_s: float, composition_s: float) -> Dict[str, Any]:
    n_samples, n_ch = size_info
    alerts = []
    for c in ctx.get("alerts", []):
        rec = _alert_record(c)
        if rec["dv_m3"] is None and "mogi" in c:
            rec["dv_m3"] = float(c["mogi"]["dv_rate"] * ctx["span_years"])
        alerts.append(rec)
    return {
        "plan": plan,
        "actions": [n for n in plan if n in CONDITIONAL_ACTIONS],
        "alerts": alerts,
        "n_alerts": len(alerts),
        "node_cpu_s": float(cpu_s),
        "composition_s": float(composition_s),
        "proc_nominal_s": float(plan_processing_s(nkg, plan, n_samples, n_ch)),
        "coverage_frac": ctx.get("coverage_frac"),
        "gap_repaired": bool(ctx.get("gap_repaired", False)),
        "spatially_filtered": bool(ctx.get("spatially_filtered", False)),
        "persisted": bool(ctx.get("persisted", False)),
        "cross_validated": bool(ctx.get("cross_validated", False)),
        "hazard_volume": bool(ctx.get("hazard_volume", False)),
        "rate_up": bool(ctx.get("rate_up", False)),
        "scores": ctx.get("scores", {}),
    }


class _InSARAdapter:
    def load_context(self, feature):
        return _load_context(feature)

    def execute(self, node_id, ctx):
        return execute_node(node_id, ctx)

    def summarise(self, ctx, plan, size_info, nkg, cpu_s, composition_s):
        return _summarise(ctx, plan, size_info, nkg, cpu_s, composition_s)


_ADAPTER = _InSARAdapter()


def run_static(feature: Dict[str, Any], nkg) -> Dict[str, Any]:
    """Fixed InSAR backbone; quality and intent are never read."""
    return run_static_core(feature, nkg, _ADAPTER)


def run_igdo(feature: Dict[str, Any], nkg) -> Dict[str, Any]:
    """NKG-composed, state/intent-aware InSAR pipeline."""
    return run_igdo_core(feature, nkg, _ADAPTER)
