"""Experiment 1 (S1/S2): static vs IGDO-aware microseismic pipeline.

Sweeps SNR x completeness, with event-bearing and event-free windows, and
compares detection recall, false-alarm rate (FAR), location error and alert
latency between the fixed backbone and the NKG-composed workflow. Routing
decision accuracy (vs. the published reference policy) is reported for IGDO.

Outputs (results/):
  microseismic_results.csv   per-window raw results
  microseismic_summary.csv   per-condition aggregates
  fig_recall_far.png         recall and FAR vs SNR
  fig_locerr_latency.png     location error and alert latency vs SNR
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from synth import MSConfig, generate_window, make_station_network  # noqa: E402
from synth.microseismic import reference_signal_rms  # noqa: E402
from engine.nkg import NodeKnowledgeGraph  # noqa: E402
from engine.runners import run_igdo, run_static  # noqa: E402
from metrics import score_window, routing_score  # noqa: E402

SNR_LEVELS = [20, 10, 5, 0]          # dB
COMP_LEVELS = [1.00, 0.85, 0.70]
SEED_BASE = 20260421


def _seed(snr_i, comp_i, is_event, w):
    return (SEED_BASE + snr_i * 100000 + comp_i * 10000
            + (0 if is_event else 5000) + w)


def run(n_event: int, n_noise: int, outdir: Path) -> pd.DataFrame:
    cfg = MSConfig()
    stations = make_station_network(cfg, seed=0)
    ref_rms = reference_signal_rms(cfg, stations)
    nkg = NodeKnowledgeGraph.from_yaml(
        str(ROOT / "configs" / "nkg" / "nodes_microseismic.yaml"))

    rows = []
    for si, snr in enumerate(SNR_LEVELS):
        for ci, comp in enumerate(COMP_LEVELS):
            for is_event, nwin in ((True, n_event), (False, n_noise)):
                for w in range(nwin):
                    rng = np.random.default_rng(_seed(si, ci, is_event, w))
                    feat, truth = generate_window(
                        cfg, stations, float(snr), float(comp),
                        is_event, rng, ref_rms, window_index=w)
                    for pipe, runner in (("static", run_static),
                                         ("igdo", run_igdo)):
                        res = runner(feat, nkg)
                        sc = score_window(res, truth)
                        row = {"snr_db": snr, "completeness": comp,
                               "window_type": "event" if is_event else "noise",
                               "window": w, "pipeline": pipe}
                        row.update(sc)
                        if pipe == "igdo":
                            row.update(routing_score(res["actions"], truth))
                        else:
                            row.update({"routing_exact": np.nan,
                                        "routing_hamming": np.nan,
                                        "expected_actions": "",
                                        "actual_actions": ""})
                        rows.append(row)
    df = pd.DataFrame(rows)
    outdir.mkdir(parents=True, exist_ok=True)
    df.to_csv(outdir / "microseismic_results.csv", index=False)

    # ---- aggregate ----
    ev = df[df.window_type == "event"]
    nz = df[df.window_type == "noise"]
    summary = (
        ev.groupby(["snr_db", "completeness", "pipeline"])
        .agg(recall=("correct_detect", "mean"),
             recall_3d=("correct_detect_3d", "mean"),
             detection_rate=("detected_any", "mean"),
             horiz_err_med_m=("horiz_error_m", "median"),
             depth_err_med_m=("depth_error_m", "median"),
             err3d_med_m=("min_loc_error_m", "median"),
             horiz_err_p90_m=("horiz_error_m", lambda x: np.nanpercentile(x, 90)),
             latency_med_s=("alert_latency_s", "median"),
             composition_med_ms=("composition_s", lambda x: 1000 * np.median(x)),
             cpu_med_ms=("node_cpu_s", lambda x: 1000 * np.median(x)),
             routing_exact=("routing_exact", "mean"),
             routing_hamming=("routing_hamming", "mean"))
        .reset_index())
    far = (nz.groupby(["snr_db", "completeness", "pipeline"])
             .agg(FAR=("false_alarm", "mean"),
                  false_alarms=("n_alerts", "sum"))
             .reset_index())
    summary = summary.merge(far, on=["snr_db", "completeness", "pipeline"])
    summary.to_csv(outdir / "microseismic_summary.csv", index=False)
    return summary


def plot(summary: pd.DataFrame, outdir: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    comps = sorted(summary.completeness.unique())
    styles = {1.00: ("o", "-"), 0.85: ("s", "--"), 0.70: ("^", ":")}
    colors = {"static": "#4477AA", "igdo": "#EE6677"}

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for comp in comps:
        marker, ls = styles.get(comp, ("o", "-"))
        for pipe in ("static", "igdo"):
            d = summary[(summary.completeness == comp) &
                        (summary.pipeline == pipe)].sort_values("snr_db",
                                                                ascending=False)
            axes[0].plot(d.snr_db, d.recall, marker=marker, ls=ls,
                         color=colors[pipe],
                         label=f"{pipe}, C={comp:.2f}")
            axes[1].plot(d.snr_db, d.FAR, marker=marker, ls=ls,
                         color=colors[pipe],
                         label=f"{pipe}, C={comp:.2f}")
    axes[0].set(xlabel="expected SNR (dB)", ylabel="recall (epicentre within 150 m)",
                ylim=(-0.03, 1.05), title="Event detection recall")
    axes[1].set(xlabel="expected SNR (dB)", ylabel="false-alarm rate",
                ylim=(-0.03, 1.05), title="False-alarm rate (event-free windows)")
    for ax in axes:
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(outdir / "fig_recall_far.png", dpi=200)
    plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    for comp in comps:
        marker, ls = styles.get(comp, ("o", "-"))
        for pipe in ("static", "igdo"):
            d = summary[(summary.completeness == comp) &
                        (summary.pipeline == pipe)].sort_values("snr_db",
                                                                ascending=False)
            axes[0, 0].plot(d.snr_db, d.horiz_err_med_m, marker=marker, ls=ls,
                            color=colors[pipe], label=f"{pipe}, C={comp:.2f}")
            axes[0, 1].plot(d.snr_db, d.depth_err_med_m, marker=marker, ls=ls,
                            color=colors[pipe], label=f"{pipe}, C={comp:.2f}")
            axes[1, 0].plot(d.snr_db, d.latency_med_s, marker=marker, ls=ls,
                            color=colors[pipe], label=f"{pipe}, C={comp:.2f}")
    # routing accuracy: IGDO only
    dr = summary[summary.pipeline == "igdo"]
    for comp in comps:
        marker, _ = styles.get(comp, ("o", "-"))
        d = dr[dr.completeness == comp].sort_values("snr_db", ascending=False)
        axes[1, 1].plot(d.snr_db, d.routing_exact, marker=marker,
                        label=f"IGDO C={comp:.2f}")
    axes[0, 0].set(xlabel="expected SNR (dB)", ylabel="median horizontal error (m)",
                   title="Horizontal location error")
    axes[0, 1].set(xlabel="expected SNR (dB)", ylabel="median depth error (m)",
                   title="Depth error")
    axes[1, 0].set(xlabel="expected SNR (dB)", ylabel="median alert latency (s)",
                   title="Alert latency (nominal cost model)")
    axes[1, 1].set(xlabel="expected SNR (dB)", ylabel="routing exact-match rate",
                   ylim=(-0.03, 1.05), title="Routing decision accuracy (IGDO)")
    for ax in axes.flat:
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(outdir / "fig_locerr_latency.png", dpi=200)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-event", type=int, default=30)
    ap.add_argument("--n-noise", type=int, default=30)
    ap.add_argument("--outdir", type=str, default=str(ROOT / "results"))
    args = ap.parse_args()
    outdir = Path(args.outdir)
    summary = run(args.n_event, args.n_noise, outdir)
    plot(summary, outdir)
    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", 30)
    print(summary.to_string(index=False))
    print(f"\nResults written to {outdir}")


if __name__ == "__main__":
    main()
