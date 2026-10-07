"""Microseismic domain adapter and run_static/run_igdo entry points.

The generic execution lives in :mod:`engine.core`; this module only supplies
the microseismic binding (waveform context loader, node dispatch, summary).
The public run_static/run_igdo signatures are unchanged.
"""
from __future__ import annotations

from typing import Any, Dict, List, Tuple

import numpy as np

from gpjson41.parser import GPJSONParser
from gpjson41 import policies_microseismic  # noqa: F401  (registers 'seismic')
from nodes.ms_nodes import execute_node
from .cost_model import plan_processing_s
from .core import run_igdo_core, run_static_core

CONDITIONAL_ACTIONS = {
    "gap_repair", "event_rate_up", "robust_denoise", "cross_validate"
}


def _load_context(feature: Dict[str, Any]):
    p = GPJSONParser(feature)
    wf = p.load_role_array("waveform")
    geo = p.dataset_by_role("station_geometry")
    stations = np.array([row[1:4] for row in geo["source"]], dtype=np.float64)
    n_ch = wf.shape[0]
    ctx = {
        "wf": wf.astype(np.float64).copy(),
        "fs": float(feature["properties"]["fs"]),
        "vp": float(feature["properties"]["vp"]),
        "stations": stations,
        "good_mask": np.ones(n_ch, dtype=bool),
        "wf_raw": wf.astype(np.float64).copy(),  # unconditioned, onset refinement
        "params": {},
        "state": None,
        "intent": None,
        "triggers": None,
        "candidates": [],
        "detections": [],
        "alerts": [],
    }
    size_info = (wf.shape[1], n_ch)
    return ctx, p, size_info


def _summarise(ctx: Dict[str, Any], plan: List[str], size_info: Tuple[int, int],
               nkg, cpu_s: float, composition_s: float) -> Dict[str, Any]:
    n_samples, n_ch = size_info
    alerts = []
    for c in ctx.get("alerts", []):
        alerts.append({
            "alert_sample": int(max(c["onsets"].values())),
            "xyz": None if c["_loc"]["xyz"] is None else np.asarray(
                c["_loc"]["xyz"], dtype=float),
            "loc_rms_samples": float(c["_loc"]["rms"]),
            "n_stations": int(c["_loc"]["n"]),
        })
    return {
        "plan": plan,
        "actions": [n for n in plan if n in CONDITIONAL_ACTIONS],
        "alerts": alerts,
        "n_alerts": len(alerts),
        "node_cpu_s": float(cpu_s),
        "composition_s": float(composition_s),
        "proc_nominal_s": float(plan_processing_s(nkg, plan, n_samples, n_ch)),
        "chunk_s": float(ctx["params"].get("chunk_s", 1.0)),
        "gap_repair_found": int(ctx.get("gap_repair_found", 0)),
        "detect_band": ctx.get("detect_band", ""),
        "rate_up": bool(ctx.get("rate_up", False)),
        "scores": ctx.get("scores", {}),
    }


class _MicroseismicAdapter:
    def load_context(self, feature):
        return _load_context(feature)

    def execute(self, node_id, ctx):
        return execute_node(node_id, ctx)

    def summarise(self, ctx, plan, size_info, nkg, cpu_s, composition_s):
        return _summarise(ctx, plan, size_info, nkg, cpu_s, composition_s)


_ADAPTER = _MicroseismicAdapter()


def run_static(feature: Dict[str, Any], nkg) -> Dict[str, Any]:
    """Fixed microseismic backbone; quality and intent are never read."""
    return run_static_core(feature, nkg, _ADAPTER)


def run_igdo(feature: Dict[str, Any], nkg) -> Dict[str, Any]:
    """NKG-composed, state/intent-aware microseismic pipeline."""
    return run_igdo_core(feature, nkg, _ADAPTER)
