"""Diagnostic: calibrate quality estimation, CV residual separability, trends.

Run directly (not imported):  python experiments/diagnose.py
"""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from synth import MSConfig, generate_window, make_station_network  # noqa: E402
from synth.microseismic import reference_signal_rms, glitch_rate  # noqa: E402
from engine.nkg import NodeKnowledgeGraph  # noqa: E402
from engine.runners import run_igdo, run_static, _load_context  # noqa: E402
from nodes import ms_nodes  # noqa: E402


def _raw_candidates(feature, nkg, narrow=False):
    """Run conditioning + detection only and return located candidates."""
    ctx, _, n_ch, n_samp = _load_context(feature)
    ctx["params"] = dict(nkg.default_params)
    if narrow:
        ms_nodes.node_robust_denoise(ctx)
    else:
        ms_nodes.node_bandpass_wide(ctx)
    ms_nodes.node_sta_lta_detect(ctx)
    out = []
    for c in ctx["candidates"]:
        c["_fs"] = ctx["fs"]
        loc = ms_nodes._grid_locate(c, ctx["stations"], ctx["vp"])
        out.append((c["t_earliest"], loc["rms"], loc["n"]))
    return out


def _pct(x, q):
    return float(np.percentile(x, q)) if x else float("nan")


def main(nwin: int = 20):
    nkg = NodeKnowledgeGraph.from_yaml(
        str(ROOT / "configs/nkg/nodes_microseismic.yaml"))
    cfg = MSConfig()
    stations = make_station_network(cfg, seed=0)
    ref = reference_signal_rms(cfg, stations)
    print(f"reference signal rms = {ref:.4f}; glitch rates: "
          + ", ".join(f"{s}dB->{glitch_rate(s):.2f}" for s in (20, 10, 5, 0)))

    for snr in (20, 10, 5, 0):
        for comp in (1.0, 0.85, 0.70):
            ests, true_rms, glitch_rms = [], [], []
            st_fa, ig_fa, st_det, ig_det = 0, 0, 0, 0
            for w in range(nwin):
                rng = np.random.default_rng(
                    70000 + snr * 100 + int(comp * 100) + w)
                # event window
                fe, te = generate_window(cfg, stations, float(snr),
                                         float(comp), True, rng, ref,
                                         window_index=w)
                ests.append(fe["properties"]["pipeline"]["health"]
                            ["estSnrDb"])
                rs, ri = run_static(fe, nkg), run_igdo(fe, nkg)
                st_det += rs["n_alerts"] > 0
                ig_det += ri["n_alerts"] > 0
                for t, rms, n in _raw_candidates(fe, nkg):
                    if abs(t - te["event_sample"]) < 250 and rms == rms:
                        true_rms.append(rms)
                # noise window
                rng2 = np.random.default_rng(
                    90000 + snr * 100 + int(comp * 100) + w)
                fn, tn = generate_window(cfg, stations, float(snr),
                                         float(comp), False, rng2, ref,
                                         window_index=w)
                rsn, rin = run_static(fn, nkg), run_igdo(fn, nkg)
                st_fa += rsn["n_alerts"] > 0
                ig_fa += rin["n_alerts"] > 0
                for t, rms, n in _raw_candidates(fn, nkg):
                    if rms == rms:
                        glitch_rms.append(rms)
            print(f"\nSNR={snr:2d} C={comp:.2f} | "
                  f"estSNR med={np.median(ests):6.2f} (true {snr})")
            print(f"  event recall(n={nwin}): "
                  f"static={st_det/nwin:.2f} igdo={ig_det/nwin:.2f}")
            print(f"  noise FAR  (n={nwin}): "
                  f"static={st_fa/nwin:.2f} igdo={ig_fa/nwin:.2f}")
            print(f"  true-event loc rms samples: med={_pct(true_rms,50):5.1f}"
                  f" p90={_pct(true_rms,90):5.1f} n_pts={len(true_rms)}")
            print(f"  glitch     loc rms samples: med={_pct(glitch_rms,50):5.1f}"
                  f" p10={_pct(glitch_rms,10):5.1f} n_pts={len(glitch_rms)}")


if __name__ == "__main__":
    main()
