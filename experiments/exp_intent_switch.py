"""Experiment 3: intent-driven dynamic re-planning (cross-modality InSAR).

The SAME observed scene (fixed seed, high coherence, one Mogi source) is
submitted under three operational intents. The GPJSON 4.1 document schema is
identical in every case -- only the values of the thin, optional
``orch_intent`` block (goal, qos.priority, latency budget, accuracy class)
change -- yet the Data Engine composes a different pipeline. This directly
answers "does the intent representation need to change across application
scenarios?": the schema does not; the selected chain does, deterministically.

Scenarios:
  deformation_trend_mapping  normal, daily budget   -> fixed backbone only
  anomaly_confirmation       high,   hourly budget  -> + rate-up, persistence,
                                                        Mogi confirm+inversion
  twin_assimilation          normal, daily budget   -> + persistence, Mogi
                                                        confirm+inversion (no
                                                        emergency rate-up)

Outputs (results/):
  intent_switch.csv        per-scenario composed chain and capabilities
  fig_intent_switch.png    chain length / nominal latency / capability flags
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import jsonschema  # noqa: E402

from synth.insar import InSARConfig, generate_scene  # noqa: E402
from engine.nkg import NodeKnowledgeGraph  # noqa: E402
from engine.runners_insar import run_igdo  # noqa: E402
from gpjson41.parser import GPJSONParser  # noqa: E402
from gpjson41 import orch_profile  # noqa: E402
from engine.planner import make_plan  # noqa: E402

SCENARIOS = [
    {"scenario": "trend_mapping", "goal": "deformation_trend_mapping",
     "priority": "normal", "budget_s": 86400.0},
    {"scenario": "anomaly_confirm", "goal": "anomaly_confirmation",
     "priority": "high", "budget_s": 3600.0},
    {"scenario": "twin_assimilation", "goal": "twin_assimilation",
     "priority": "normal", "budget_s": 86400.0},
]


def run(outdir: Path) -> pd.DataFrame:
    cfg = InSARConfig()
    nkg = NodeKnowledgeGraph.from_yaml(
        str(ROOT / "configs" / "nkg" / "nodes_insar.yaml"))
    schema = json.loads((ROOT / "gpjson41" / "schema_feat.json").read_text(
        encoding="utf-8"))
    rng = np.random.default_rng(20261005)
    rows = []
    features = {}
    for sc in SCENARIOS:
        feat, truth = generate_scene(
            cfg, 0.85, True, rng, 0,
            goal=sc["goal"], priority=sc["priority"],
            latency_budget_s=sc["budget_s"])
        jsonschema.validate(feat, schema)   # identical schema across intents
        features[sc["scenario"]] = feat

        parser = GPJSONParser(feat)
        state = orch_profile.state_view(parser)
        intent = orch_profile.intent_view(parser)
        plan_obj = make_plan(nkg, state, intent)
        res = run_igdo(feat, nkg)
        rows.append({
            "scenario": sc["scenario"],
            "goal": sc["goal"],
            "priority": sc["priority"],
            "latency_budget_s": sc["budget_s"],
            "accuracy_class":
                feat["properties"]["orch_intent"]["constraints"].get(
                    "accuracyClass"),
            "n_nodes": len(res["plan"]),
            "plan": " -> ".join(res["plan"]),
            "actions": ";".join(res["actions"]),
            "rate_up": bool(res["rate_up"]),
            "spatially_filtered": bool(res["spatially_filtered"]),
            "persisted": bool(res["persisted"]),
            "cross_validated": bool(res["cross_validated"]),
            "hazard_volume": bool(res["hazard_volume"]),
            "n_alerts": res["n_alerts"],
            "confirmed": any(a["confirmed"] for a in res["alerts"]),
            "composition_ms": 1000.0 * res["composition_s"],
            "proc_nominal_s": res["proc_nominal_s"],
            "planner_score": plan_obj.get("scores", {}).get("total", np.nan),
        })
    df = pd.DataFrame(rows)
    outdir.mkdir(parents=True, exist_ok=True)
    df.to_csv(outdir / "intent_switch.csv", index=False)
    return df


def plot(df: pd.DataFrame, outdir: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = df.scenario.tolist()
    x = np.arange(len(df))
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.0))
    axes[0].bar(x, df.n_nodes, color="#4477AA")
    axes[0].set(title="composed chain length (nodes)", xticks=x,
                xticklabels=labels, ylabel="nodes")
    bars = axes[1].bar(x, df.proc_nominal_s, color="#EE6677")
    pmax = max(float(df.proc_nominal_s.max()), 1e-9)
    axes[1].set_ylim(0, pmax * 1.35)
    for xi, v in zip(x, df.proc_nominal_s):
        axes[1].text(xi, v + pmax * 0.02, f"{v:.2g}", ha="center",
                     fontsize=8)
    axes[1].text(0.02, 0.95,
                 "latency budget 3600-86400 s\n(off-scale; processing << budget)",
                 transform=axes[1].transAxes, va="top", fontsize=8)
    axes[1].set(title="nominal processing time (s)", xticks=x,
                xticklabels=labels, ylabel="seconds")
    flags = ["rate_up", "persisted", "cross_validated", "hazard_volume"]
    bottom = np.zeros(len(df))
    for i, fl in enumerate(flags):
        vals = df[fl].astype(float).to_numpy()
        axes[2].bar(x, vals, bottom=bottom, label=fl,
                    color=["#228833", "#66CCEE", "#CCBB44", "#EE6677"][i])
        bottom += vals
    axes[2].set(title="activated capabilities", xticks=x,
                xticklabels=labels, ylim=(0, 4.6))
    axes[2].legend(fontsize=7, ncol=2)
    for ax in axes:
        ax.grid(alpha=0.3, axis="y")
        ax.tick_params(axis="x", labelrotation=20)
    fig.tight_layout()
    fig.savefig(outdir / "fig_intent_switch.png", dpi=200)
    plt.close(fig)


def main():
    outdir = ROOT / "results"
    df = run(outdir)
    plot(df, outdir)
    pd.set_option("display.width", 240)
    pd.set_option("display.max_columns", 30)
    for _, r in df.iterrows():
        print(f"\n[{r.scenario}] goal={r.goal} priority={r.priority} "
              f"budget={r.latency_budget_s:.0f}s")
        print("  chain:", r.plan)
        print(f"  rate_up={r.rate_up} persisted={r.persisted} "
              f"cross_validated={r.cross_validated} hazard={r.hazard_volume} "
              f"confirmed={r.confirmed}")
    print(f"\nResults written to {outdir}")


if __name__ == "__main__":
    main()
