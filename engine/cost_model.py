"""Nominal execution cost model for end-to-end latency simulation."""
from __future__ import annotations

from typing import List

from .nkg import NodeKnowledgeGraph


def plan_processing_s(nkg: NodeKnowledgeGraph, plan: List[str],
                      n_samples: int, n_ch: int) -> float:
    """Sum of nominal node latencies (seconds).

    These are published engineering-style nominal costs (per-node base +
    per-sample/channel coefficients), NOT measured wall clock. The experiment
    additionally records measured CPU time; the paper reports the nominal model
    for end-to-end alert latency and the measured values for composition time.
    """
    return sum(nkg.cost_ms(nid, n_samples, n_ch) for nid in plan) / 1000.0
