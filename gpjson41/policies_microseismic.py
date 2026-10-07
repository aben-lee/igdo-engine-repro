"""Microseismic domain reference policy.

Maps microseismic ingest estimates onto the GPJSON 4.1 core quality block and
defines the auditable reference routing policy for the S1/S2 experiment. The
policy is consumed (a) conceptually by the NKG-driven planner and (b) as the
ground truth for microseismic routing-decision accuracy.

Registration: the ``seismic.*`` kind namespace maps to this policy.
"""
from __future__ import annotations

from typing import Any, Dict, List

import numpy as np

from . import orch_profile as core

# --- action vocabulary (must match the microseismic NKG node ids) ----------
ACT_DENOISE = "robust_denoise"
ACT_GAP_REPAIR = "gap_repair"
ACT_CROSS_VALIDATE = "cross_validate"
ACT_RATE_UP = "event_rate_up"

ACTION_UNIVERSE = [ACT_GAP_REPAIR, ACT_DENOISE, ACT_CROSS_VALIDATE, ACT_RATE_UP]

# --- microseismic-specific thresholds (published in the appendix) ----------
SNR_LOW_DB = 8.0           # below this, narrow-band screening is warranted


def est_snr_from_state(state: Dict[str, Any]) -> float:
    """Recover the ingest-side SNR estimate stashed in health (dB)."""
    return float(state["health"].get("estSnrDb", 20.0))


def quality_from_estimates(
    est_snr_db: float,
    n_good: int,
    n_total: int,
    n_flagged: int,
) -> Dict[str, float]:
    """Deterministic, auditable mapping from ingest estimates to core quality."""
    completeness = n_good / max(1, n_total)
    s = 1.0 / (1.0 + np.exp(-(est_snr_db - 6.0) / 3.0))
    confidence = float(min(1.0, max(0.0, s * (0.75 + 0.25 * completeness))))
    outlier_rate = float(min(1.0, n_flagged / max(1, n_total)))
    validity = core.validity_from_confidence(confidence)
    return {
        "completeness": float(completeness),
        "outlierRate": outlier_rate,
        "confidence": confidence,
        "validity": validity,
        "estSnrDb": float(est_snr_db),
    }


def reference_actions(state: Dict[str, Any], intent: Dict[str, Any]) -> List[str]:
    """Microseismic expert rule set; unconditional backbone is implied."""
    q = state["quality"]
    snr = est_snr_from_state(state)
    actions: List[str] = []

    degraded = (snr < SNR_LOW_DB or q["confidence"] < core.CONF_LOW
                or q["completeness"] < core.COMPLETENESS_MIN)
    incomplete = q["completeness"] < core.COMPLETENESS_MIN
    if incomplete:
        actions.append(ACT_GAP_REPAIR)
    if degraded:
        actions.append(ACT_DENOISE)
        # any adaptive front-end is paired with strict downstream confirmation
        actions.append(ACT_CROSS_VALIDATE)

    if (core.high_priority(intent)
            and core.goal_contains(intent, "near_boundary")):
        actions.append(ACT_RATE_UP)
    return actions


class _Policy:
    ACTION_UNIVERSE = ACTION_UNIVERSE
    reference_actions = staticmethod(reference_actions)


core.register_domain("seismic", _Policy)
