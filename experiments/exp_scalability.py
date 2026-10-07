"""Experiment 2: workflow composition-time scalability.

Measures planner wall-clock for three strategies on random layered NKG DAGs:
  * exhaustive   - enumerate all feasible paths and score every one;
  * pruned       - hard-constraint pruning + beam search (beam=20);
  * static       - one fixed workflow, O(1) lookup;
and reports the pruned-vs-exhaustive optimality gap (multi-objective score).

Outputs (results/):
  scalability_results.csv
  fig_scalability.png
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from engine.planner import time_planners  # noqa: E402

NODE_COUNTS = [10, 25, 50, 100, 200]


def run(n_tasks: int, outdir: Path) -> pd.DataFrame:
    rows = []
    for n in NODE_COUNTS:
        rows.append(time_planners(n, n_tasks, seed=20260421 + n))
    df = pd.DataFrame(rows)
    outdir.mkdir(parents=True, exist_ok=True)
    df.to_csv(outdir / "scalability_results.csv", index=False)
    return df


def plot(df: pd.DataFrame, outdir: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    x = df.n_nodes
    axes[0].plot(x, df.exhaustive_ms_median, "o-", label="exhaustive median")
    axes[0].fill_between(x, df.exhaustive_ms_median, df.exhaustive_ms_p95,
                         alpha=0.15, label="exhaustive P95")
    axes[0].plot(x, df.pruned_ms_median, "s-", label="pruned beam=20 median")
    axes[0].fill_between(x, df.pruned_ms_median, df.pruned_ms_p95, alpha=0.15)
    axes[0].plot(x, df.static_ms_median, "^--", label="static lookup")
    axes[0].set(xlabel="number of registered nodes",
                ylabel="composition time (ms)", yscale="log",
                title="Workflow composition time")
    axes[0].grid(alpha=0.3, which="both")
    axes[0].legend(fontsize=8)

    axes[1].plot(x, df.optimality_gap, "D-", color="#228833")
    axes[1].set(xlabel="number of registered nodes",
                ylabel="mean |score gap| vs exhaustive",
                title="Pruned-search optimality gap")
    axes[1].grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(outdir / "fig_scalability.png", dpi=200)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-tasks", type=int, default=100)
    ap.add_argument("--outdir", type=str, default=str(ROOT / "results"))
    args = ap.parse_args()
    outdir = Path(args.outdir)
    df = run(args.n_tasks, outdir)
    plot(df, outdir)
    pd.set_option("display.width", 200)
    print(df.to_string(index=False))
    print(f"\nResults written to {outdir}")


if __name__ == "__main__":
    main()
