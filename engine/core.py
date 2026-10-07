"""Modal-agnostic pipeline execution core.

A *domain adapter* binds the generic engine to one modality's payload:

    adapter.load_context(feature) -> (ctx, parser, size_info)
    adapter.execute(node_id, ctx) -> nominal/measured node cost (seconds)
    adapter.summarise(ctx, plan, size_info, nkg, cpu_s, composition_s) -> dict

The static core never reads state/intent; the IGDO core reads the common
state/intent vectors and asks the (data-driven) planner to compose the chain.
Microseismic and InSAR share this exact core; only the adapter differs.
"""
from __future__ import annotations

import time
from typing import Any, Dict, Tuple

from gpjson41 import orch_profile
from .planner import make_plan


def run_static_core(feature: Dict[str, Any], nkg, adapter) -> Dict[str, Any]:
    """Fixed backbone; quality and intent are never read."""
    ctx, _parser, size_info = adapter.load_context(feature)
    ctx["params"] = dict(nkg.default_params)
    plan = list(nkg.backbone)
    cpu_s = 0.0
    for nid in plan:
        cpu_s += adapter.execute(nid, ctx)
    return adapter.summarise(ctx, plan, size_info, nkg, cpu_s, 0.0)


def run_igdo_core(feature: Dict[str, Any], nkg, adapter) -> Dict[str, Any]:
    """State/intent are read and the workflow is composed by the Data Engine."""
    ctx, parser, size_info = adapter.load_context(feature)
    state = orch_profile.state_view(parser)
    intent = orch_profile.intent_view(parser)
    ctx["state"], ctx["intent"] = state, intent

    t0 = time.perf_counter()
    plan_obj = make_plan(nkg, state, intent)
    composition_s = time.perf_counter() - t0
    ctx["params"] = plan_obj["params"]
    ctx["scores"] = plan_obj["scores"]

    cpu_s = 0.0
    for nid in plan_obj["nodes"]:
        cpu_s += adapter.execute(nid, ctx)
    result = adapter.summarise(ctx, plan_obj["nodes"], size_info, nkg,
                               cpu_s, composition_s)
    result["reasons"] = plan_obj["reasons"]
    return result
