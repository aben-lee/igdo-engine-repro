"""Node Knowledge Graph: loader, precondition evaluation, chain enumeration."""
from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

import numpy as np
import yaml


def _get_field(ctx: Dict[str, Any], field: str) -> Any:
    """Resolve dotted paths against a flat state/intent evaluation context."""
    cur: Any = ctx
    for part in field.split("."):
        if isinstance(cur, dict) and part in cur:
            cur = cur[part]
        else:
            return None
    return cur


_OPS = {
    "<": lambda a, b: a is not None and a < b,
    ">": lambda a, b: a is not None and a > b,
    "<=": lambda a, b: a is not None and a <= b,
    ">=": lambda a, b: a is not None and a >= b,
    "==": lambda a, b: a == b,
    "!=": lambda a, b: a != b,
    "contains": lambda a, b: a is not None and b in a,
}


def eval_clause(clause: Dict[str, Any], ctx: Dict[str, Any]) -> bool:
    a = _get_field(ctx, clause["field"])
    op = _OPS[clause["op"]]
    return bool(op(a, clause["value"]))


def eval_precondition(pre: Optional[Dict[str, Any]], ctx: Dict[str, Any]) -> bool:
    if not pre:
        return True
    if "all" in pre:
        return all(eval_clause(c, ctx) for c in pre["all"])
    if "any" in pre:
        return any(eval_clause(c, ctx) for c in pre["any"])
    return eval_clause(pre, ctx)


def make_eval_context(state: Dict[str, Any], intent: Dict[str, Any]) -> Dict[str, Any]:
    """Flatten state/intent into the namespace used by NKG preconditions."""
    q = state.get("quality", {})
    health = state.get("health", {})
    qos = intent.get("constraints", {}).get("qos", {})
    # pass the WHOLE modality-specific health block through (microseismic
    # estSnrDb, InSAR meanCoherence, ...); defaults keep old domains working.
    health_ctx = {"sensorStatus": "OK", "estSnrDb": 20.0}
    health_ctx.update(health)
    return {
        "quality": {"completeness": q.get("completeness", 1.0),
                    "outlierRate": q.get("outlierRate", 0.0),
                    "confidence": q.get("confidence", 1.0),
                    "validity": q.get("validity", "valid")},
        "health": health_ctx,
        "stage": state.get("stage", "raw"),
        "qos": {"priority": qos.get("priority", "normal")},
        "intent": {"goal": intent.get("goal", "")},
    }


class NodeKnowledgeGraph:
    def __init__(self, spec: Dict[str, Any]):
        self.spec = spec
        self.nodes: Dict[str, Dict[str, Any]] = {n["id"]: n for n in spec["nodes"]}
        self.backbone = list(spec.get("backbone", []))
        self.conditional_order = list(spec.get("conditional_order", []))
        self.default_params = dict(spec.get("default_params", {}))

    @classmethod
    def from_yaml(cls, path: str) -> "NodeKnowledgeGraph":
        with open(path, "r", encoding="utf-8") as fh:
            return cls(yaml.safe_load(fh))

    def applicable_nodes(self, ctx: Dict[str, Any]) -> List[str]:
        out = []
        for nid in self.conditional_order:
            node = self.nodes[nid]
            if eval_precondition(node.get("preconditions"), ctx):
                out.append(nid)
        return out

    def cost_ms(self, nid: str, n_samples: int, n_ch: int) -> float:
        c = self.nodes[nid]["nominal_cost_ms"]
        return float(c["base"] + c["per_k_sample_ch"] * n_samples * n_ch / 1000.0)
