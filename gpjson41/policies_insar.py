"""InSAR domain reference policy.

Maps InSAR coherence/coverage estimates onto the GPJSON 4.1 core quality
block and defines the auditable reference routing policy for the Mogi
cross-modality experiment. It is structurally identical in role to the
microseismic policy (quality routing + adaptive front-end + physics-based
back-end confirmation + intent-driven scheduling), only the indicators and
operators change.

Registration: the ``insar.*`` kind namespace maps to this policy.

Thresholds are initial values to be calibrated by diagnostics (see
experiments/exp_insar.py); they are physical and never tuned to a target
score.
"""
from __future__ import annotations

from typing import Any, Dict, List

import numpy as np

from . import orch_profile as core

# --- action vocabulary (must match the InSAR NKG node ids) -----------------
ACT_GAP_REPAIR = "gap_repair_insar"
ACT_SPATIAL_FILTER = "spatial_filter"
ACT_PERSIST = "temporal_persist_check"
ACT_MOGI_CV = "mogi_cross_validate"
ACT_RATE_UP = "investigation_rate_up"

ACTION_UNIVERSE = [ACT_GAP_REPAIR, ACT_SPATIAL_FILTER, ACT_PERSIST,
                   ACT_MOGI_CV, ACT_RATE_UP]

# --- InSAR-specific thresholds (initial, diagnostics-calibrated) -----------
COH_GATE = 0.35          # coherence gate: pixels below are masked/unreliable
MEAN_COH_LOW = 0.75      # below this mean coherence, adaptive filtering is used
COH_MID = 0.55           # sigmoid midpoint for the confidence mapping
COH_SLOPE = 0.12         # sigmoid scale

# Cramer-Rao interferometric phase std (sigma_phi [rad]) and its LOS mapping.
#   sigma_phi = (1/sqrt(2 L)) * sqrt(1-gamma^2)/gamma,  L = number of looks
#   sigma_los [mm] = sigma_phi / (2 pi) * (lambda/2), Sentinel-1 lambda=56 mm
N_LOOKS = 20.0
WAVELENGTH_MM = 56.0


def phase_sigma_mm(coherence: np.ndarray) -> np.ndarray:
    """Per-pixel LOS phase-noise std (mm) as a function of coherence."""
    g = np.clip(np.asarray(coherence, dtype=np.float64), 0.05, 0.999)
    sigma_phi = (1.0 / np.sqrt(2.0 * N_LOOKS)) * np.sqrt(1.0 - g ** 2) / g
    return sigma_phi / (2.0 * np.pi) * (WAVELENGTH_MM / 2.0)


def quality_from_coherence(mean_gamma: float, coverage: float
                           ) -> Dict[str, float]:
    """Deterministic, auditable mapping from coherence to core quality."""
    coverage = float(np.clip(coverage, 0.0, 1.0))
    s = 1.0 / (1.0 + np.exp(-(mean_gamma - COH_MID) / COH_SLOPE))
    confidence = float(min(1.0, max(0.0, s * (0.70 + 0.30 * coverage))))
    outlier_rate = 1.0 - coverage
    validity = core.validity_from_confidence(confidence)
    return {
        "completeness": coverage,
        "outlierRate": float(outlier_rate),
        "confidence": confidence,
        "validity": validity,
        "meanCoherence": float(mean_gamma),
    }


def reference_actions(state: Dict[str, Any], intent: Dict[str, Any]) -> List[str]:
    """InSAR expert rule set; the unconditional backbone is implied."""
    q = state["quality"]
    h = state.get("health", {})
    mean_coh = float(h.get("meanCoherence", 1.0))

    incomplete = q["completeness"] < core.COMPLETENESS_MIN
    low_coh = (mean_coh < MEAN_COH_LOW
               or q["outlierRate"] > core.OUTLIER_HIGH
               or q["confidence"] < core.CONF_LOW)
    # both anomaly confirmation and digital-twin assimilation require a
    # temporally/persistently checked, physics-confirmed source; only the
    # former (at high priority) requests denser acquisition.
    confirm = (core.goal_contains(intent, "anomaly_confirmation")
               or core.goal_contains(intent, "twin_assimilation"))

    actions: List[str] = []
    if incomplete:
        actions.append(ACT_GAP_REPAIR)
    if low_coh:
        actions.append(ACT_SPATIAL_FILTER)
    # any adaptive front-end, or an explicit confirmation/assimilation goal, is
    # paired with temporal persistence and a physics-based (Mogi) confirmation
    if incomplete or low_coh or confirm:
        actions.append(ACT_PERSIST)
        actions.append(ACT_MOGI_CV)
    if core.goal_contains(intent, "anomaly_confirmation") and core.high_priority(intent):
        actions.append(ACT_RATE_UP)
    return actions


class _Policy:
    ACTION_UNIVERSE = ACTION_UNIVERSE
    reference_actions = staticmethod(reference_actions)


core.register_domain("insar", _Policy)
