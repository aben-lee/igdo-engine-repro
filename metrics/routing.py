"""Microseismic routing decision accuracy vs. the domain reference policy."""
from __future__ import annotations

from typing import Any, Dict, List

from gpjson41 import policies_microseismic as pol

ACTION_UNIVERSE = pol.ACTION_UNIVERSE


def expected_actions(truth: Dict[str, Any]) -> List[str]:
    """What the microseismic reference policy would do under GROUND-TRUTH state."""
    n_total = len(truth["stations"])
    n_bad = len(truth["dead_channels"]) + len(truth["noisy_channels"])
    n_good = n_total - n_bad
    q = pol.quality_from_estimates(truth["true_snr_db"], n_good, n_total, n_bad)
    state = {
        "quality": {"completeness": q["completeness"],
                    "outlierRate": q["outlierRate"],
                    "confidence": q["confidence"],
                    "validity": q["validity"]},
        "health": {"estSnrDb": float(truth["true_snr_db"])},
        "stage": "raw",
    }
    goal = "rockburst_event_detection"
    if truth.get("near_boundary"):
        goal += "_near_boundary"
    intent = {"goal": goal, "outcomes": [],
              "constraints": {"qos": {"priority": truth.get("priority", "normal")}}}
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
