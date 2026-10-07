# IGDO-Engine — Minimal Reproducible Experiment Package

Reproducible Python subset of the IGDO-Engine orchestration kernel described in
**"Autonomous Geophysical Monitoring for Infrastructure Digital Twins with
Intelligent Geo-data and Self-organizing Pipelines"** (GRSM-2026-00141.R1).

It provides the quantitative evidence requested by reviewers R2/R3/R4:

* **S1/S2 microseismic experiment** — fixed (static) pipeline vs. the
  NKG-composed, state/intent-aware pipeline, over controlled SNR and
  channel-completeness conditions;
* **cross-modality InSAR Mogi experiment** — the *same* GPJSON 4.1 protocol,
  state/intent vocabulary, planner and execution kernel driving a structurally
  different payload (a `Grid2D` multi-epoch InSAR LOS stack rather than a
  waveform gather), showing unified representation across heterogeneous
  geophysical observations;
* **intent-driven re-planning experiment** — one unchanged schema, three
  operational intents, three deterministically different composed chains;
* **workflow-planning scalability experiment** — composition time of
  exhaustive / pruned / static planners vs. number of registered nodes;
* a **legal GPJSON 4.1** protocol layer, with the paper's state/intent
  semantics implemented as a thin *optional extension profile* (`orch_intent`,
  mapped onto core `quality` + the Pipeline Profile). GPJSON is **kept at
  version 4.1** (no 5.x bump): per §12.2 of the specification, adding
  `properties` fields, roles and kinds is forward-compatible and requires no
  version change, whereas the v4.0-draft State & Intent Profile was withdrawn
  in 4.1 in favour of core `quality` + the Pipeline Profile. We validate that
  decision and add only the one genuinely missing token (`goal`/`outcomes`).

> The production platform is C++/Qt (EgeoCoreLib `GPjson/`, GPjsonBuilder /
> GPjsonParser / EgeoFeature) with Python processing nodes. This package is the
> **minimal reproducible subset of its orchestration kernel** and deliberately
> has no Qt dependency.

## 1. Install

```bash
python -m pip install -r requirements.txt
```

Dependencies: numpy, scipy, pandas, matplotlib, networkx, PyYAML, jsonschema,
obspy. Requires Python 3.10+ (developed and tested on Python 3.14).

## 2. Reproduce everything

```bash
python experiments/run_all.py
```

or run the experiments separately:

```bash
python experiments/exp_microseismic.py --n-event 30 --n-noise 30
python experiments/exp_scalability.py  --n-tasks 100
python experiments/exp_insar.py        --n-src 25 --n-nosrc 25
python experiments/exp_intent_switch.py
python -m scripts.gen_examples          # write the two example Features
```

Smoke tests (protocol round-trip, schema validation, detection, planner rules
for **both** modalities, and intent switching):

```bash
python -m tests.test_smoke
```

All randomness is seeded; results are deterministic.

## 3. Outputs (`results/`)

| File | Content |
|---|---|
| `microseismic_results.csv` | per-window raw outcomes (both pipelines) |
| `microseismic_summary.csv` | recall / detection rate / FAR / location error / latency / routing accuracy per condition |
| `fig_recall_far.png` | recall and false-alarm rate vs SNR |
| `fig_locerr_latency.png` | horizontal/depth location error, alert latency, routing accuracy vs SNR |
| `insar_results.csv` | per-scene InSAR outcomes (both pipelines) |
| `insar_summary.csv` | recall (60/90/120 m) / FAR / horizontal, depth, volume error / fit NRSS / routing per coherence |
| `fig_insar_recall_far.png` | Mogi source recall and false-alarm rate vs coherence |
| `fig_insar_source.png` | inverted source location/depth/volume error, fit quality, routing vs coherence |
| `intent_switch.csv`, `fig_intent_switch.png` | composed chain and capabilities for three intents |
| `scalability_results.csv` | planner composition time (median/P95) and optimality gap |
| `fig_scalability.png` | composition time vs number of nodes |

Two self-contained, schema-valid GPJSON 4.1 Features are written to
`examples/` (`microseismic_example.json`, `insar_example.json`): different
geometry/payload/kind, identical core `quality`, Pipeline Profile and
`orch_intent` blocks.

## 3b. Scalability findings

On random layered NKG DAGs (10-200 nodes, 100 planning tasks each): exhaustive
enumeration grows super-linearly (median composition time 0.01 ms at 10 nodes
to 12.5 ms at 200 nodes; P95 61 ms), beam search (beam=20) stays near-linear
(4.2 ms median / 10 ms P95 at 200 nodes, a ~3-6× speed-up), and static lookup
is constant (~0.3 µs). The measured optimality gap is **zero** for this
benchmark: the score is additive over layers, so retaining the top-20 partial
prefixes per layer preserves the globally optimal path. This is a property of
the layered, additive scoring structure used here, not a general guarantee —
under cross-node coupling constraints a non-zero gap is expected, which is why
both the gap and the beam width are reported rather than assumed.

## 4. Experiment design (S1/S2)

* Array: 8 stations in a 3-D deep-mine geometry — six roof/floor/rib stations
  along two roadways plus two down-hole geophones for vertical resolution;
  local mine grid; `fs = 1000 Hz`, 5 s windows.
* Source: Ricker wavelet (`fp = 100 Hz`), homogeneous `vp = 4500 m/s`,
  geometric spreading `1/r`; hypocentre and origin time randomised.
* Noise: stationary AR(1) coloured noise (`rho = 0.85`), calibrated so a
  reference event would meet the target SNR — **identical noise floors in
  event-bearing and event-free windows** — plus realistic non-seismic
  **transients** whose rate rises with noise level: multi-channel near-
  simultaneous low-frequency (15-45 Hz) mechanical vibration and single/
  two-channel broad-band knocks. These transients are the dominant false-alarm
  source; a stationary coloured-noise-only model produces essentially zero
  false alarms for an STA/LTA detector and cannot test any pipeline.
* Conditions: expected SNR `{20, 10, 5, 0}` dB × completeness `{1.00, 0.85,
  0.70}`; 30 event + 30 event-free Monte-Carlo windows per condition.
* Bad channels are half dead (flat), half noisy; ingest estimates quality
  **from the data** (noise-only leader + channel-power checks), never from
  ground truth.

### Pipelines

* **static**: fixed `bandpass_wide (20-300 Hz) → STA/LTA → refine_onsets →
  grid-locate → alert`, thresholds and cadence never change;
* **igdo**: the Data Engine reads state/intent, evaluates NKG preconditions and
  composes the workflow. Under any degradation (expected SNR < 8 dB, confidence
  < 0.60, or completeness < 0.90) it switches the front end to the event band
  (`robust_denoise`: adaptive 60-200 Hz narrow-band screening, which rejects
  out-of-band mechanical vibration without distorting the impulsive arrival),
  runs `gap_repair` on low completeness (bad channels diagnosed from the
  noise-only leader and excluded), and adds velocity-model `cross_validate`
  (≥3 stations, arrival-time residual within 25 samples, hypocentre inside the
  monitored volume); `event_rate_up` handles high-priority near-boundary goals.

**Detection band and location band are separated.** Screening may run on a
narrow band for interference rejection, but after detection every candidate's
onsets are re-picked on the wide-banded **raw** waveform by an AIC refinement
node (`refine_onsets`, present in *both* pipelines) before grid location. This
keeps narrow-band interference rejection from degrading arrival-time accuracy,
and makes the location comparison fair — both pipelines locate from identical
refined onsets.

Both call the **same node implementations**; only composition differs. With
healthy data (SNR ≥ 10 dB, complete array) the IGDO plan reduces to the
backbone, so the two pipelines are identical by construction — the adaptive
behaviour activates only where the data warrant it.

### Headline results (30 event + 30 event-free windows per condition)

* **Degraded regime (SNR ≤ 5 dB, or incomplete array):** IGDO cuts the
  false-alarm rate from 0.30-0.73 (static) to 0-0.07 while keeping recall
  0.87-1.00 — equal to or slightly above static; e.g. at 0 dB / complete array
  FAR falls from 0.73 to 0.07 with recall 0.97 vs 1.00.
* **Healthy regime (SNR ≥ 10 dB, complete array):** the two pipelines coincide
  exactly (same FAR, same per-window locations), as designed.
* **Location accuracy:** for the IGDO chain the median epicentral error is
  ≈ 9.7-13.6 m and median depth error ≈ 6.4-14.2 m across the reported
  conditions (well inside the 150 m recall tolerance), with no penalty
  relative to static (the worst static medians reach ≈ 17 m in the most
  degraded cell).
* **Routing accuracy:** exact-set match 0.97-1.00 against the published
  reference policy (the single non-unity cell is the SNR ≈ 10 dB boundary,
  where ingest SNR estimates straddle the 8 dB threshold).
* **Honest boundary:** at 10 dB with a complete array both pipelines keep
  FAR ≈ 0.40. Data above the quality threshold are not re-routed, so an
  occasional in-band transient is not suppressed at this Phase-2 *pre-routing*
  stage; suppressing such transients needs the run-time closed loop (Phase 3).
* **Latency:** at the same 1 s cadence the two pipelines differ by < 0.1 s in
  S1/S2; the latency benefit of adaptive cadence is deferred to S3.
* **Composition cost:** median workflow-composition time is 0.17-0.23 ms for
  the 9-node microseismic NKG (see the scalability experiment).

### Metrics

* **recall** — event window with an alert whose **epicentral (horizontal)**
  hypocentre is within 150 m of truth (microseismic convention; a 3-D recall is
  also reported). Horizontal and depth location errors are reported separately
  because depth is intrinsically weaker for a limited-aperture array;
* **detection rate** — fraction of event windows raising any alert;
* **FAR** — fraction of event-free windows raising an alert;
* **alert latency** — chunk-aligned wait for the final contributing arrival
  plus nominal node processing cost (a published nominal cost model; measured
  CPU time is also recorded). At the same 1 s cadence the two pipelines have
  comparable latency in S1/S2 — the IGDO benefit there is FAR/recall, while the
  latency gain of `event_rate_up` belongs to the S3 near-boundary scenario;
* **routing decision accuracy** — IGDO actions vs. the published reference
  policy evaluated under ground-truth conditions (exact-set match and per-action
  Hamming score). This replaces the previously unsupported "rule-triggering
  accuracy above 90%" statement with a reproducible number.

## 4b. Cross-modality experiment: InSAR Mogi deformation (S4/S5)

This experiment answers the objection that microseismics alone cannot
demonstrate a *unified representation of heterogeneous geophysical
observations*: InSAR deformation fields and microseismic gathers have
fundamentally different data structures and processing. The same GPJSON 4.1
objects, the same common state/intent vocabulary, the same data-driven planner
and the same modal-agnostic execution core (`engine/core.py`) drive both; only
the loader, the domain policy and the NKG change.

### Unified representation across two modalities

| Aspect | microseismic (`seismic.trace`) | InSAR (`insar.deformation`) |
|---|---|---|
| geometry | 8 point stations (`point3d`) | one `Grid2D` (64×64 @ 30 m) |
| payload | `(station, sample)` waveform gather | `(epoch, y, x)` LOS stack + coherence cube + LOS unit vector |
| primary health indicator | `estSnrDb` | `meanCoherence`, `coverage`, `atmPhaseRmsMm`, `demResidualMm` |
| `quality.completeness` | fraction of live channels | valid-pixel coverage |
| `quality.outlierRate` | bad-channel fraction | `1 − coverage` |
| `quality.confidence` | SNR sigmoid | coherence sigmoid (`quality_from_coherence`) |
| adaptive front-end | `robust_denoise` (narrow-band) | `spatial_filter` (edge-preserving) + `gap_repair_insar` |
| temporal confirmation | velocity-model `cross_validate` | `temporal_persist_check` (multi-looked linear R²) |
| physical back-end | grid hypocentre inversion | coarse-to-fine **Mogi** inversion (x, y, d, ΔV) |
| emergency scheduling | `event_rate_up` | `investigation_rate_up` |
| unchanged | core `quality` five fields, Pipeline Profile (`stage/qos/health/...`), `orch_intent{goal,outcomes,constraints}`, planner, execution core | identical |

`insar.deformation` is a new registered kind in the `insar.*` namespace
(Appendix G style); registering a kind is forward-compatible under GPJSON 4.1
§12.2 and needs no version change.

### Scene model (fully analytic, no external data)

* `Grid2D` 64×64 pixels at 30 m, T = 16 epochs every 12 days; single
  descending track (incidence 34°, heading 102°) with the LOS unit vector.
* A **Mogi (1958)** point pressure source in a homogeneous elastic half-space
  (`u_r = C r/(r²+d²)^{3/2}`, `u_z = C d/(r²+d²)^{3/2}`,
  `C = (1−ν)/π · ΔV`), sited at a stable-scatterer island; displacement is
  linear in ΔV, so inversion grid-searches (x, y, d) and solves ΔV by least
  squares. Depth 300-1100 m, peak LOS 12-40 mm.
* Dominant false anomaly: a **temporally independent**, long-wavelength
  tropospheric power-law field (1.2 km correlation) whose RMS grows as
  coherence drops; plus clustered decorrelation voids, a static long-wave DEM
  residual, and per-pixel **Cramér-Rao** phase noise
  `σ_φ = (1/√(2L))·√(1−γ²)/γ`. Two persistent-scatterer islands exist in
  *both* source and source-free scenes; source-free scenes use the identical
  noise floor (no truth leakage). Quality is estimated from the data
  (coherence/coverage), never from truth.
* Conditions: mean coherence `{0.85, 0.65, 0.45, 0.30}`, 25 source-bearing +
  25 source-free scenes each; the intent is fixed to
  `anomaly_confirmation` at normal priority so coherence is the single
  manipulated factor (the high-priority rate-up is exercised in §4c).

### Pipelines

* **static** backbone: `coh_mask → per_pixel_rate (per-pixel linear-in-time
  LOS slope) → threshold_detect (rate gate + connected clusters, centroid) →
  alert`; thresholds never change and no source physics is fitted.
* **igdo** composes from state/intent: `gap_repair_insar` when coverage < 0.90;
  `spatial_filter` (90 m edge-preserving) at low coherence/outliers;
  `temporal_persist_check` fits a linear trend to the **spatially multi-looked
  core** deformation time series (SBAS-style: coherent signal adds in phase,
  short-correlation noise decays as 1/√N, an atmospheric patch shows no
  trend); `mogi_cross_validate` fits the concentric Mogi morphology and keeps
  a candidate only if the normalised residual is below gate; `mogi_inversion`
  then returns (x, y, d, ΔV); `investigation_rate_up` fires only for a
  high-priority confirmation goal. Both pipelines call the same node code.

### Headline results (25 + 25 scenes per coherence level)

* **False alarms:** IGDO FAR is **0 at every coherence level**; the static
  pipeline FAR is 1.00 at 0.65/0.45/0.30 (89/130/121 false alarms across 25
  scenes) and 0 only at 0.85. The persistence + Mogi-morphology tests remove
  essentially all atmospheric/noise clusters.
* **High coherence (0.85):** IGDO source recall is **0.96 at 90 m** (0.76 at
  60 m, 1.00 at 120 m), median horizontal error **41 m**, median depth error
  **30 m**, median |ΔV| error **6.5 %**, reconstructed rate-field NRMSE 0.09,
  Mogi fit NRSS 0.22. The static centroid is biased to ~189 m and yields no
  depth/volume at all.
* **Routing:** exact-set match against the InSAR reference policy is **1.00**
  at every coherence level.
* **Graceful, conservative degradation:** as coherence falls, IGDO trades
  recall for its zero-FAR guarantee (detection 0.84/0.48/0.64 at 0.65/0.45/
  0.30) and source-parameter uncertainty grows monotonically (median
  horizontal 135-219 m, depth 112-242 m, |ΔV| 14-40 %, NRSS 0.30-0.41). The
  static pipeline keeps "alarming" (FAR = 1) with 180-277 m scatter, i.e. its
  nominal recall at low coherence is meaningless.
* **Cost:** workflow composition is 0.2-0.27 ms; the Mogi inversion adds a
  median 0.5-0.6 s CPU at high/medium coherence — negligible against a
  daily latency budget.

### Honest physical limitations

Single-track LOS Mogi inversion has an intrinsic **depth-volume degeneracy**,
so depth is weaker than horizontal location and degrades with atmospheric
noise; we report horizontal, depth and volume error separately rather than
hiding it. A planar ramp term was deliberately **not** added to the Mogi fit:
it is strongly collinear with a deep/broad Mogi field and biases the recovered
volume, so long-wave atmosphere is rejected upstream by the temporal and
morphology tests. These limits motivate exactly the multi-modality fusion the
digital twin is for (ascending/descending tracks, GNSS/levelling assimilation).

## 4c. Intent-driven dynamic re-planning

The same high-coherence source scene is submitted under three intents; the
GPJSON schema is identical (all three validate), only the `orch_intent`
*values* change:

| scenario | goal / priority / budget | composed chain |
|---|---|---|
| trend mapping | `deformation_trend_mapping` / normal / 86400 s | `coh_mask → per_pixel_rate → threshold_detect → alert` |
| anomaly confirmation | `anomaly_confirmation` / **high** / 3600 s | `investigation_rate_up → … → temporal_persist_check → mogi_cross_validate → mogi_inversion → alert` |
| twin assimilation | `twin_assimilation` / normal / 86400 s | `… → temporal_persist_check → mogi_cross_validate → mogi_inversion → alert` (no emergency rate-up) |

Thus the intent **representation** (schema/fields) does not change across
scenarios, while the selected chain, scheduling token and delivered products
change deterministically with goal, priority and latency budget — the direct
answer to whether intent must "dynamically change" per application scenario.

## 5. Mapping to review comments

| Comment | Addressed by |
|---|---|
| R2-3 / R3-2 / R4-1: no runnable prototype or quantitative results | full package + CSV/figures |
| R2-2 / R3-3: why not extend OGC SensorThings / state-intent not in v4.1 | legal GPJSON 4.1 objects + `orch_intent` thin extension, mapping in `gpjson41/orch_profile.py` |
| R2-5: multi-objective Data Engine decision mechanism | `engine/planner.py` (hard constraints + weighted score) |
| R3-4: unsupported "90%" claim | routing accuracy against an explicit reference policy |
| R2-4: NKG construction/maintenance | `configs/nkg/nodes_microseismic.yaml` + `engine/nkg.py` |
| scalability of self-organising pipelines | `experiments/exp_scalability.py` |
| microseismics cannot represent heterogeneous data; InSAR has a different structure; how are unified state/intent fields defined per data type? | §4b, `synth/insar.py`, `nodes/insar_nodes.py`, `gpjson41/policies_insar.py`, `engine/runners_insar.py`, two `examples/*.json`, mapping table |
| must intent change across application scenarios? | §4c, `experiments/exp_intent_switch.py` (one schema, three chains) |
| should GPJSON add a State & Intent profile / bump version? | kept at **4.1**; thin forward-compatible `orch_intent` on core `quality` + Pipeline Profile (§12.2); see `gpjson41/orch_profile.py` |

## 6. Layout

```
igdo-engine-repro/
├─ gpjson41/          # v4.1 builder/parser, JSON Schema, modality-agnostic
│                     #   orch core + policies_microseismic / policies_insar
├─ synth/             # microseismic and InSAR (Mogi) synthetic generators
├─ nodes/             # ms_nodes.py and insar_nodes.py processing nodes
├─ engine/            # NKG, cost model, planner, modal-agnostic core.py,
│                     #   microseismic adapter (runners.py) + InSAR adapter
│                     #   (runners_insar.py)
├─ metrics/           # detection/routing metrics for both modalities
├─ configs/nkg/       # node knowledge graphs (nodes_microseismic/insar.yaml)
├─ experiments/       # four experiments, run_all.py, threshold diagnostics
├─ scripts/           # gen_examples.py
├─ examples/          # two self-contained schema-valid GPJSON 4.1 Features
├─ tests/             # smoke tests (both modalities + intent switching)
└─ results/           # CSV + figures (generated)
```

## 7. Scope / non-goals of this batch

This package covers S1/S2 microseismics, the S4/S5 InSAR Mogi cross-modality
experiment, intent-driven re-planning and planning scalability. Reserved for a
subsequent batch (interfaces kept compatible): real SAR / PS-SBAS ingestion,
ascending/descending fusion and ERA5/GACOS atmospheric correction, the IoT
time-series experiment (S6), the end-to-end S3 near-boundary rate-up scenario,
and other real-data sample ingestion.

## 8. Code availability, archival and citation

The exact version used for the manuscript is permanently archived as a **code**
record on **Science Data Bank (ScienceDB)** and mirrored on GitHub:

* Archived release **v1.0.0**, DOI: `https://doi.org/10.57760/sciencedb.014tw`
* CSTR: `https://cstr.cn/31253.11.sciencedb.014tw`
* Source repository: `https://github.com/aben-lee/igdo-engine-repro`

At submission, ScienceDB creates a private read-only link that can be shared
with reviewers during peer review; the GitHub repository is already public. The
code is released under the BSD 3-Clause License (see `LICENSE`). Please cite both
the archived code (ScienceDB DOI) and the article; citation metadata is in
`CITATION.cff`.
