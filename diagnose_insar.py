"""Diagnostics / physical self-checks for the synthetic InSAR generator.

Run:  python diagnose_insar.py
Checks Mogi analytic properties (axisymmetry, peak above source, linearity in
dV, LOS peak control) and prints coherence/quality/noise-floor diagnostics by
coherence level. Threshold calibration for the NKG lives in exp_insar.py.
"""
from __future__ import annotations

import math

import numpy as np

from synth.insar import (
    InSARConfig,
    _power_law_field,
    c_dv_from_peak_los,
    coherence_stack,
    generate_scene,
    grid_coordinates,
    los_unit_vector,
    mogi_displacement,
)


def check_mogi(cfg: InSARConfig) -> None:
    x, y, X, Y = grid_coordinates(cfg)
    cx, cy = cfg.nx // 2, cfg.ny // 2
    sx, sy = float(x[cx]), float(y[cy])
    depth, c_dv = 800.0, 12000.0
    ux, uy, uz = mogi_displacement(X, Y, sx, sy, depth, c_dv)

    # peak is directly above the source and equals c/d^2 (vertical)
    iy, ix = np.unravel_index(np.argmax(uz), uz.shape)
    assert (ix, iy) == (cx, cy), "Mogi vertical peak not above source"
    assert abs(uz[iy, ix] - c_dv / depth ** 2) < 1e-9, "Mogi peak magnitude"
    assert abs(ux[iy, ix]) < 1e-9 and abs(uy[iy, ix]) < 1e-9, "zero horiz at top"

    # axisymmetry: uz on constant-radius rings (grid-discretised tolerance)
    rr = np.sqrt((np.arange(cfg.ny)[:, None] - cy) ** 2
                 + (np.arange(cfg.nx)[None, :] - cx) ** 2)
    for rpx in (5, 10, 20):
        ring = np.abs(rr - rpx) < 0.6
        v = uz[ring]
        rel = v.std() / abs(v.mean())
        assert rel < 0.03, f"axisymmetry violated at r={rpx}: {rel}"

    # linear in dV
    _, _, uz2 = mogi_displacement(X, Y, sx, sy, depth, 2.0 * c_dv)
    assert np.allclose(uz2, 2.0 * uz), "Mogi not linear in dV"

    # LOS peak control via the inverse relation
    n = los_unit_vector(cfg)
    cdv = c_dv_from_peak_los(cfg, depth, 20.0, n)
    _, _, uzp = mogi_displacement(X, Y, sx, sy, depth, cdv)
    peak = 1000.0 * (n[2] * uzp[iy, ix])
    assert abs(peak - 20.0) < 1e-4, f"LOS peak control off: {peak}"
    print("[ok] Mogi: peak above source, axisymmetric, linear in dV, "
          "LOS peak controlled")


def check_atmosphere_decorrelation(cfg: InSARConfig) -> None:
    # independent power-law fields; over a finite (correlation-length-scale)
    # field of view a single pair has noisy sample correlation, so average
    # many independent pairs.
    corrs = []
    for seed in range(20):
        rng = np.random.default_rng(seed)
        f1 = _power_law_field(cfg.ny, cfg.nx, cfg.spacing_m, cfg.atm_corr_m,
                              cfg.atm_beta, rng)
        f2 = _power_law_field(cfg.ny, cfg.nx, cfg.spacing_m, cfg.atm_corr_m,
                              cfg.atm_beta, rng)
        corrs.append(float(np.corrcoef(f1.ravel(), f2.ravel())[0, 1]))
    mean_abs = float(np.mean(np.abs(corrs)))
    assert mean_abs < 0.12, f"atmospheric fields not independent: {mean_abs}"
    print(f"[ok] mean |epoch-to-epoch atmospheric correlation| = {mean_abs:.3f} (~0)")


def diagnose_levels(n_scene: int = 20) -> None:
    cfg = InSARConfig()
    print(f"\ngrid {cfg.nx}x{cfg.ny} @ {cfg.spacing_m} m, T={cfg.n_epochs}, "
          f"span={cfg.span_years:.3f} yr, LOS n={los_unit_vector(cfg).round(3)}")
    header = (f"{'meanCoh':>8} {'scene':>6} {'meanCohEst':>11} {'coverage':>9} "
              f"{'conf':>6} {'valid':>8} {'atmCfg':>7} {'atmEst':>7} "
              f"{'demEst':>7}")
    for mc in (0.85, 0.65, 0.45, 0.30):
        est_cov, est_conf, est_atm_s, est_atm_n, valid = [], [], [], [], []
        est_g = []
        for k in range(n_scene):
            rng = np.random.default_rng(1000 + int(mc * 100) * 1000 + k)
            fs, ts = generate_scene(cfg, mc, True, rng, scene_index=k)
            fn, tn = generate_scene(cfg, mc, False,
                                    np.random.default_rng(
                                        5000 + int(mc * 100) * 1000 + k),
                                    scene_index=k)
            est_g.append(ts["mean_coh_est"])
            est_cov.append(ts["coverage"])
            est_conf.append(fs["properties"]["quality"]["confidence"])
            valid.append(fs["properties"]["quality"]["validity"])
            est_atm_s.append(ts["atm_rms_est_mm"])
            est_atm_n.append(tn["atm_rms_est_mm"])
        from collections import Counter
        print(header)
        print(f"{mc:>8} {'src':>6} {np.mean(est_g):>11.3f} "
              f"{np.mean(est_cov):>9.3f} {np.mean(est_conf):>6.3f} "
              f"{dict(Counter(valid))!s:>8} {cfg.atm_rms_mm_at_085 if mc==0.85 else 0:>7}"
              f" {np.mean(est_atm_s):>7.2f}")
        print(f"{'':>8} {'no-src':>6} {'':>11} {'':>9} {'':>6} {'':>8} "
              f"{'':>7} {np.mean(est_atm_n):>7.2f}")
        floor_ratio = np.mean(est_atm_s) / max(1e-9, np.mean(est_atm_n))
        print(f"       atm RMS source/no-source ratio = {floor_ratio:.3f} "
              f"(want ~1)  atmCfg={_atm_cfg(cfg, mc):.1f} mm")


def _atm_cfg(cfg, mc):
    from synth.insar import atm_rms_mm
    return atm_rms_mm(mc, cfg)


def main() -> None:
    cfg = InSARConfig()
    check_mogi(cfg)
    check_atmosphere_decorrelation(cfg)
    diagnose_levels()
    print("\nAll physical self-checks passed.")


if __name__ == "__main__":
    main()
