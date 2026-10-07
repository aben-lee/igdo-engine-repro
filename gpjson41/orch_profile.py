"""Modal-agnostic orchestration core: mapping IGDO semantics onto GPJSON 4.1.

This module is the single source of truth for the *optional orchestration
profile over GPJSON 4.1* described in the revised manuscript (Section II-C).
It is deliberately modality-independent:

    paper state.quality      <-> properties.quality (core v4.1)
    paper state.uncertainty  <-> properties.quality.uncertainty
    paper state.stage        <-> properties.pipeline.stage.level
    paper state.health       <-> properties.pipeline.health (free, kind-scoped)
    paper intent.constraints <-> properties.pipeline.latencyBudget / qos
    paper intent.goal        <-> properties.orch_intent.goal (optional)

The *unified* layer is a fixed vocabulary (core quality + pipeline profile +
goal/outcomes). Modality-specific health indicators (microseismic estSnrDb,
InSAR meanCoherence, ...) live in pipeline.health and are interpreted by
per-domain reference policies, registered here and selected by the Feature's
``kind`` namespace. This realises the paper's "fixed common state/intent
dictionary + kind-scoped health extension + per-domain NKG" design.

Domain policies live in ``policies_microseismic`` / ``policies_insar`` and
register themselves on import.
"""
from __future__ import annotations

from typing import Any, Dict, List

from .parser import GPJSONParser

# --- core, modality-independent quality gates -----------------------------
CONF_LOW = 0.60            # core quality confidence gate
COMPLETENESS_MIN = 0.90    # below this, gaps must be handled
OUTLIER_HIGH = 0.20        # above this, robust conditioning is warranted

# --- domain reference-policy registry --------------------------------------
_POLICIES: Dict[str, Any] = {}


def register_domain(kind_prefix: str, policy: Any) -> None:
    """Register a domain policy (microseismic, insar, ...) by kind prefix."""
    _POLICIES[kind_prefix.lower()] = policy


def _prefix(kind: str) -> str:
    return str(kind).lower().split(".")[0] if kind else ""


def _policy_for(kind: str) -> Any:
    p = _POLICIES.get(_prefix(kind))
    if p is None:
        raise KeyError(
            f"No orchestration policy registered for kind {kind!r} "
            f"(prefix {_prefix(kind)!r}); import the matching policies_* "
            f"module or register one via register_domain().")
    return p


def reference_actions(kind: str, state: Dict[str, Any],
                      intent: Dict[str, Any]) -> List[str]:
    """Dispatch to the domain reference policy selected by ``kind``."""
    return _policy_for(kind).reference_actions(state, intent)


def action_universe(kind: str) -> List[str]:
    """The auditable conditional-action vocabulary for a domain."""
    return list(_policy_for(kind).ACTION_UNIVERSE)


# --- modality-independent state/intent projection -------------------------
def state_view(p: GPJSONParser) -> Dict[str, Any]:
    """Project a v4.1 Feature onto the paper's (common) state vector."""
    return {
        "quality": {
            "completeness": p.completeness,
            "outlierRate": p.outlier_rate,
            "confidence": p.confidence,
            "validity": p.validity,
        },
        "uncertainty": p.uncertainty,
        "stage": p.stage,
        "health": p.health,
    }


def intent_view(p: GPJSONParser) -> Dict[str, Any]:
    """Project a v4.1 Feature onto the paper's (common) intent vector."""
    constraints = dict(p.intent.get("constraints", {}))
    if p.latency_budget is not None:
        constraints.setdefault("latencyBudget_s", p.latency_budget)
    if p.qos:
        constraints.setdefault("qos", p.qos)
    return {
        "goal": p.goal,
        "outcomes": list(p.intent.get("outcomes", [])),
        "constraints": constraints,
    }


# --- shared, auditable helpers ---------------------------------------------
def validity_from_confidence(conf: float) -> str:
    if conf >= CONF_LOW:
        return "valid"
    return "suspect" if conf >= 0.35 else "invalid"


def core_quality_degraded(q: Dict[str, Any]) -> bool:
    """Modality-independent degradation from the CORE quality block only."""
    return (q.get("confidence", 1.0) < CONF_LOW
            or q.get("completeness", 1.0) < COMPLETENESS_MIN
            or q.get("outlierRate", 0.0) > OUTLIER_HIGH)


def high_priority(intent: Dict[str, Any]) -> bool:
    return intent.get("constraints", {}).get("qos", {}).get("priority") == "high"


def goal_contains(intent: Dict[str, Any], token: str) -> bool:
    return token in intent.get("goal", "")
