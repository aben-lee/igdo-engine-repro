"""Experiment 2 (cross-modality, InSAR): static vs IGDO Mogi monitoring.

Single manipulated factor = interferometric coherence (proxy for atmospheric
path delay + phase noise + coverage); the intent is fixed to the operational
task "anomaly_confirmation" at normal priority (so the rate-up scheduling
token, which requires high priority, is exercised separately in
exp_intent_switch.py). For each coherence level we generate source-bearing
(single Mogi point source) and source-free scenes, and compare:

  * detection recall at 60/90/120 m horizontal tolerance,
  * false-alarm rate on source-free scenes,
  * horizontal / depth / volume inversion accuracy (IGDO),
  * reconstructed rate-field NRMSE and Mogi fit NRSS,
  * routing decision accuracy vs. the InSAR reference policy.

Outputs (results/):
  insar_results.csv   per-scene raw results
  insar_summary.csv   per-coherence aggregates
  fig_insar_recall_far.png   recall and FAR vs coherence
  fig_insar_source.png       inversion source-parameter errors vs coherence
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from synth.insar import InSARConfig, generate_scene  # noqa: E402
from engine.nkg import NodeKnowledgeGraph  # noqa: E402
from engine.runners_insar import run_igdo, run_static  # noqa: E402
from metrics import score_scene, routing_score_insar  # noqa: E402

COH_LEVELS = [0.85, 0.65, 0.45, 0.30]
SEED_BASE = 20261005
GOAL = "anomaly_confirmation"
PRIORITY = "normal"
BUDGET_S = 86400.0


def _seed(ci, is_source, w):
    return SEED_BASE + ci * 10000 + (0 if is_source else 5000) + w


def run(n_src: int, n_nosrc: int, outdir: Path) -> pd.DataFrame:
    cfg = InSARConfig()
    nkg = NodeKnowledgeGraph.from_yaml(
        str(ROOT / "configs" / "nkg" / "nodes_insar.yaml"))
    rows = []
    for ci, mc in enumerate(COH_LEVELS):
        for is_source, nsc in ((True, n_src), (False, n_nosrc)):
            for w in range(nsc):
                rng = np.random.default_rng(_seed(ci, is_source, w))
                feat, truth = generate_scene(
                    cfg, float(mc), is_source, rng, w,
                    goal=GOAL, priority=PRIORITY, latency_budget_s=BUDGET_S)
                for pipe, runner in (("static", run_static),
                                     ("igdo", run_igdo)):
                    res = runner(feat, nkg)
                    sc = score_scene(res, truth)
                    row = {"mean_coh": mc,
                           "scene_type": "source" if is_source else "nosource",
                           "scene": w, "pipeline": pipe}
                    row.update(sc)
                    if pipe == "igdo":
                        row.update(routing_score_insar(res["actions"], truth))
                    else:
                        row.update({"routing_exact": np.nan,
                                    "routing_hamming": np.nan,
                                    "expected_actions": "",
                                    "actual_actions": ""})
                    rows.append(row)
        print(f"coherence {mc} done")
    df = pd.DataFrame(rows)
    outdir.mkdir(parents=True, exist_ok=True)
    df.to_csv(outdir / "insar_results.csv", index=False)

    src = df[df.scene_type == "source"]
    nos = df[df.scene_type == "nosource"]
    summary = (
        src.groupby(["mean_coh", "pipeline"])
        .agg(detection_rate=("detected_any", "mean"),
             confirmed_rate=("confirmed", "mean"),
             recall_60=("correct_60m", "mean"),
             recall_90=("correct_90m", "mean"),
             recall_120=("correct_120m", "mean"),
             horiz_err_med_m=("horiz_error_m", "median"),
             horiz_err_p90_m=("horiz_error_m",
                              lambda x: np.nanpercentile(x, 90)),
             depth_err_med_m=("depth_error_m", "median"),
             dv_err_med_pct=("dv_rel_error_pct", "median"),
             peak_err_med_pct=("peak_rate_err_pct", "median"),
             rate_nrmse_med=("rate_field_nrmse", "median"),
             nrss_med=("nrss", "median"),
             composition_med_ms=("composition_s",
                                 lambda x: 1000 * np.nanmedian(x)),
             cpu_med_ms=("node_cpu_s", lambda x: 1000 * np.nanmedian(x)),
             routing_exact=("routing_exact", "mean"),
             routing_hamming=("routing_hamming", "mean"))
        .reset_index())
    far = (nos.groupby(["mean_coh", "pipeline"])
              .agg(FAR=("false_alarm", "mean"),
                   false_alarms=("n_alerts", "sum"))
              .reset_index())
    summary = summary.merge(far, on=["mean_coh", "pipeline"])
    summary.to_csv(outdir / "insar_summary.csv", index=False)
    return summary


def plot(summary: pd.DataFrame, outdir: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = {"static": "#4477AA", "igdo": "#EE6677"}
    coh = sorted(summary.mean_coh.unique(), reverse=True)

    def d(pipe, col):
        dd = summary[summary.pipeline == pipe].set_index("mean_coh")
        return [dd.loc[c, col] for c in coh]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for pipe in ("static", "igdo"):
        axes[0].plot(coh, d(pipe, "recall_90"), "o-", color=colors[pipe],
                     label=f"{pipe} recall@90m")
        axes[1].plot(coh, d(pipe, "FAR"), "s-", color=colors[pipe],
                     label=f"{pipe} FAR")
    # tolerance sensitivity for IGDO recall
    axes[0].plot(coh, d("igdo", "recall_60"), "^--", color="#CCBB44",
                 alpha=0.8, label="igdo recall@60m")
    axes[0].plot(coh, d("igdo", "recall_120"), "v--", color="#228833",
                 alpha=0.8, label="igdo recall@120m")
    axes[0].set(xlabel="mean coherence", ylabel="source recall",
                ylim=(-0.03, 1.05), title="Mogi source detection recall")
    axes[1].set(xlabel="mean coherence", ylabel="false-alarm rate",
                ylim=(-0.03, 1.05), title="False-alarm rate (source-free scenes)")
    for ax in axes:
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(outdir / "fig_insar_recall_far.png", dpi=200)
    plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    axes[0, 0].plot(coh, d("static", "horiz_err_med_m"), "s-",
                    color=colors["static"], label="static (centroid)")
    axes[0, 0].plot(coh, d("igdo", "horiz_err_med_m"), "o-",
                    color=colors["igdo"], label="igdo (Mogi inversion)")
    axes[0, 0].set(title="Horizontal source error (median, m)",
                   xlabel="mean coherence", ylabel="m")
    axes[0, 1].plot(coh, d("igdo", "depth_err_med_m"), "o-",
                    color=colors["igdo"])
    axes[0, 1].set(title="Source depth error, IGDO (median, m)",
                   xlabel="mean coherence", ylabel="m")
    axes[1, 0].plot(coh, d("igdo", "dv_err_med_pct"), "o-",
                    color=colors["igdo"])
    axes[1, 0].set(title="Volume change |dV| error, IGDO (median, %%)",
                   xlabel="mean coherence", ylabel="%")
    axes[1, 1].plot(coh, d("igdo", "rate_nrmse_med"), "o-", color="#228833",
                    label="rate-field NRMSE")
    axes[1, 1].plot(coh, d("igdo", "nrss_med"), "^--", color="#66CCEE",
                    label="Mogi fit NRSS")
    axes[1, 1].plot(coh, d("igdo", "routing_exact"), "s:", color="#4477AA",
                    label="routing exact-match")
    axes[1, 1].set(title="Fit quality & routing (IGDO)",
                   xlabel="mean coherence", ylabel="ratio")
    axes[1, 1].legend(fontsize=8)
    for ax in axes.flat:
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8, loc="best")
    fig.tight_layout()
    fig.savefig(outdir / "fig_insar_source.png", dpi=200)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-src", type=int, default=25)
    ap.add_argument("--n-nosrc", type=int, default=25)
    ap.add_argument("--outdir", type=str, default=str(ROOT / "results"))
    args = ap.parse_args()
    outdir = Path(args.outdir)
    summary = run(args.n_src, args.n_nosrc, outdir)
    plot(summary, outdir)
    pd.set_option("display.width", 220)
    pd.set_option("display.max_columns", 30)
    print(summary.to_string(index=False))
    print(f"\nResults written to {outdir}")


if __name__ == "__main__":
    main()
