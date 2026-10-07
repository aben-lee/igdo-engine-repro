"""Data Engine planner: hard-constraint filtering + multi-objective scoring.

Phase-2 realisation used in the study:
  1. evaluate NKG node preconditions against the IGDO state/intent;
  2. compose an ordered workflow (conditional nodes inserted into the backbone);
  3. score candidates on latency / robustness / uncertainty-awareness /
     cost / twin relevance, with weights derived from intent constraints.

The module also provides a random layered-DAG NKG generator and three planners
(exhaustive, pruned beam search, static lookup) used by the scalability
experiment (workflow composition time vs. number of registered nodes).
"""
from __future__ import annotations

import copy
import time
from typing import Any, Dict, List, Tuple

import numpy as np

from .nkg import (
    NodeKnowledgeGraph,
    eval_clause,
    eval_precondition,
    make_eval_context,
)

# --------------------------------------------------------------------------
# Domain-independent rule planner (driven entirely by the NKG specification)
# --------------------------------------------------------------------------
def assemble_plan(nkg: NodeKnowledgeGraph, ctx: Dict[str, Any]
                  ) -> Tuple[List[str], List[str]]:
    """Compose the workflow from the NKG ``conditional_order``.

    Keep every node whose preconditions hold (unconditional nodes always
    hold); a selected node suppresses the default node named in its
    ``replaces`` field (e.g. robust_denoise replaces bandpass_wide). Each
    selected conditional node contributes its optional ``reason`` string.
    """
    applicable = set(nkg.applicable_nodes(ctx))
    suppressed: set = set()
    chain: List[str] = []
    reasons: List[str] = []
    for nid in nkg.conditional_order:
        if nid not in applicable or nid in suppressed:
            continue
        chain.append(nid)
        node = nkg.nodes[nid]
        rep = node.get("replaces")
        if rep:
            suppressed.add(rep)
        reason = node.get("reason")
        if reason:
            reasons.append(reason)
    return chain, reasons


def make_plan(nkg: NodeKnowledgeGraph, state: Dict[str, Any],
              intent: Dict[str, Any]) -> Dict[str, Any]:
    ctx = make_eval_context(state, intent)
    chain, reasons = assemble_plan(nkg, ctx)
    params = copy.deepcopy(nkg.default_params)
    scores = score_chain(nkg, chain, ctx)
    return {"nodes": chain, "params": params, "reasons": reasons,
            "scores": scores, "eval_ctx": ctx}


def _weights_from_intent(intent: Dict[str, Any]) -> Dict[str, float]:
    cons = intent.get("constraints", {})
    budget = cons.get("latencyBudget_s", 4.0)
    qos = cons.get("qos", {})
    # tight budget -> more weight on latency; high priority -> on robustness
    w_lat = 0.35 if budget and budget < 2.0 else 0.20
    w_rob = 0.35 if qos.get("priority") == "high" else 0.20
    w_unc, w_cost, w_rel = 0.25, 0.10, 0.10
    total = w_lat + w_rob + w_unc + w_cost + w_rel
    return {"latency": w_lat / total, "robustness": w_rob / total,
            "uncertainty": w_unc / total, "cost": w_cost / total,
            "relevance": w_rel / total}


def score_chain(nkg: NodeKnowledgeGraph, chain: List[str],
                ctx: Dict[str, Any], n_samples: int = 5000,
                n_ch: int = 8) -> Dict[str, float]:
    from .cost_model import plan_processing_s

    scoring = nkg.spec.get("scoring", {})
    budget = float(scoring.get("budget_s", 2.0))
    protective = set(scoring.get("protective", []))
    emergency_token = scoring.get("emergency_goal_token", "")
    emergency_node = scoring.get("emergency_node", "")

    latency_s = plan_processing_s(nkg, chain, n_samples, n_ch)
    s_lat = float(max(0.0, 1.0 - latency_s / budget))
    s_rob = float(np.mean([nkg.nodes[n]["descriptors"].get("robustness", 0.5)
                           for n in chain]))
    # uncertainty awareness, derived from the NKG itself (modality-independent):
    # a protective node that is applicable must actually be in the chain
    applicable = set(nkg.applicable_nodes(ctx))
    degraded = bool(protective & applicable)
    acted = bool(protective & set(chain))
    s_unc = 1.0 if ((degraded and acted) or not degraded) else 0.0
    s_cost = float(max(0.0, 1.0 - len(chain) / 10.0))
    need_emergency = bool(emergency_token) and emergency_token in \
        ctx["intent"]["goal"]
    s_rel = 1.0 if (not need_emergency or emergency_node in chain) else 0.2
    parts = {"latency": s_lat, "robustness": s_rob, "uncertainty": s_unc,
             "cost": s_cost, "relevance": s_rel}
    intent = {"constraints": {"latencyBudget_s": None, "qos":
                              {"priority": ctx["qos"]["priority"]}},
              "goal": ctx["intent"]["goal"]}
    w = _weights_from_intent(intent)
    s = sum(w[k] * parts[k] for k in parts)
    return {"S": float(s), **{k: float(v) for k, v in parts.items()},
            "latency_s": float(latency_s)}


# --------------------------------------------------------------------------
# Random layered NKG for the scalability experiment
# --------------------------------------------------------------------------
def build_random_dag(n_nodes: int, rng: np.random.Generator):
    """Layered DAG: ingest -> k preprocess candidates -> k detect candidates
    -> locate -> alert. Each candidate carries random cost/quality descriptors
    and a random minimum-confidence precondition."""
    import networkx as nx

    g = nx.DiGraph()
    layers = [["ingest"], [], [], ["locate"], ["alert"]]
    # distribute the remaining nodes across the two candidate layers
    remaining = n_nodes - 3
    k = max(1, remaining // 2)
    pre = [f"pre_{i}" for i in range(k)]
    det = [f"det_{i}" for i in range(n_nodes - 3 - k)]
    layers[1], layers[2] = pre, det
    if not det:
        layers[1] = pre[:-1]
        layers[2] = [pre[-1]]

    def desc():
        return {"lat_ms": float(rng.uniform(2, 40)),
                "robustness": float(rng.uniform(0.4, 0.99)),
                "min_conf": float(rng.uniform(0.2, 0.9))}

    for li, layer in enumerate(layers):
        for node in layer:
            g.add_node(node, layer=li, **(desc() if node not in
                       ("ingest", "locate", "alert") else
                       {"lat_ms": 3.0, "robustness": 0.8, "min_conf": 0.0}))
    for a, b in zip(layers, layers[1:]):
        for na in a:
            for nb in b:
                g.add_edge(na, nb)
    return g, layers


def _feasible(g, node: str, conf: float) -> bool:
    return conf >= g.nodes[node].get("min_conf", 0.0)


def enumerate_paths(g, layers: List[List[str]], conf: float
                    ) -> List[List[str]]:
    paths: List[List[str]] = [[]]
    for layer in layers:
        feas = [n for n in layer if _feasible(g, n, conf)]
        if not feas:
            return []
        paths = [p + [n] for p in paths for n in feas]
    return paths


def pruned_paths(g, layers: List[List[str]], conf: float, beam: int = 20
                 ) -> List[List[str]]:
    paths: List[List[str]] = [[]]

    def _score(p):
        # partial score: prefer low latency, high robustness
        lat = sum(g.nodes[n]["lat_ms"] for n in p)
        rob = sum(g.nodes[n]["robustness"] for n in p) / max(1, len(p))
        return rob - lat / 200.0

    for layer in layers:
        feas = [n for n in layer if _feasible(g, n, conf)]
        if not feas:
            return []
        paths = [p + [n] for p in paths for n in feas]
        paths = sorted(paths, key=_score, reverse=True)[:beam]
    return paths


def _path_score(g, path: List[str]) -> float:
    lat = sum(g.nodes[n]["lat_ms"] for n in path)
    rob = sum(g.nodes[n]["robustness"] for n in path) / len(path)
    return 0.6 * rob - 0.4 * lat / 200.0


def time_planners(n_nodes: int, n_tasks: int, seed: int, beam: int = 20
                  ) -> Dict[str, Any]:
    rng = np.random.default_rng(seed)
    g, layers = build_random_dag(n_nodes, rng)
    static_path = [layer[0] for layer in layers]
    ex_t, pr_t, st_t = [], [], []
    gaps = []
    for t in range(n_tasks):
        conf = float(rng.uniform(0.1, 1.0))

        t0 = time.perf_counter()
        allp = enumerate_paths(g, layers, conf)
        best = max(allp, key=lambda p: _path_score(g, p)) if allp else None
        ex_t.append((time.perf_counter() - t0) * 1000.0)

        t0 = time.perf_counter()
        prp = pruned_paths(g, layers, conf, beam)
        pbest = max(prp, key=lambda p: _path_score(g, p)) if prp else None
        pr_t.append((time.perf_counter() - t0) * 1000.0)

        t0 = time.perf_counter()
        _ = static_path  # lookup, O(1)
        st_t.append((time.perf_counter() - t0) * 1000.0)

        if best and pbest:
            gaps.append(abs(_path_score(g, best) - _path_score(g, pbest)))

    def pct(x, q):
        return float(np.percentile(x, q))

    return {
        "n_nodes": n_nodes,
        "n_tasks": n_tasks,
        "exhaustive_ms_median": pct(ex_t, 50),
        "exhaustive_ms_p95": pct(ex_t, 95),
        "pruned_ms_median": pct(pr_t, 50),
        "pruned_ms_p95": pct(pr_t, 95),
        "static_ms_median": pct(st_t, 50),
        "static_ms_p95": pct(st_t, 95),
        "optimality_gap": float(np.mean(gaps)) if gaps else np.nan,
    }
