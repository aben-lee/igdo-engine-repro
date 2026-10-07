# GPJSON 4.1 — Protocol Specification (Companion Document)

### Geo-Pipeline JSON for geoscience sensor, model and pipeline data

**Core format:** GPJSON **4.1** (Stable)
**Optional extension described here:** Orchestration Profile **1.0** (`orch_intent` / state projection)
**Companion to:** the IGDO-Engine reproducible package and the manuscript *"Autonomous Geophysical Monitoring for Infrastructure Digital Twins with Intelligent Geo-Data and Self-organizing Pipelines"*
**Status of this document:** implementation-oriented companion. It documents (i) the stable GPJSON 4.1 core, restricted to the subset exercised by the companion code, and (ii) the optional Orchestration Profile introduced by the manuscript. All examples are synthetic and contain no proprietary, project, site or product identifiers.

> **Licensing.** The reference *code* is released under the BSD 3-Clause License (`../LICENSE`). This *specification document* is licensed under the **Creative Commons Attribution 4.0 International License (CC-BY-4.0)**.

---

## Table of contents

1. Introduction
2. Object model
3. The `kind` namespace
4. Coordinate reference systems (`crs`)
5. `properties` and the core `quality` block
6. `datasets` — payload carriers
7. Time and date
8. Security and integrity (summary)
9. Pipeline Profile (recommended)
10. **Orchestration Profile (optional extension over 4.1)**
11. Relationship to OGC standards
12. Versioning and forward compatibility
13. Reference implementation and examples
14. Normative and related references
15. Citation

---

## 1. Introduction

GPJSON (**Geo-Pipeline JSON**) is a JSON format derived from the GeoJSON format [RFC 7946], designed for geoscience sensor data, geological digital-twin models, monitoring time series, and node-based data pipelines. It preserves the GeoJSON `Feature` / `FeatureCollection` / `Geometry` structure and adds three things:

1. **`datasets`** — structured payload carriers attached to a `Feature` (tables, N-dimensional arrays, embedded binaries, external URIs, streaming segments);
2. **Extended geometry types** — eight 3-D / grid / point-cloud / voxel geometries in addition to the seven standard GeoJSON geometries;
3. **`kind` semantic classification plus optional profiles** — business-semantic typing and pipeline/orchestration metadata carried with the data, so that a data object is self-describing, routable and auditable.

A GPJSON Feature that carries neither `datasets` nor an extended geometry is a valid, standard GeoJSON Feature.

### 1.1 Conformance keywords

The key words **MUST**, **MUST NOT**, **REQUIRED**, **SHALL**, **SHALL NOT**, **SHOULD**, **SHOULD NOT**, **RECOMMENDED**, **MAY**, and **OPTIONAL** are to be interpreted as in [RFC 2119].

### 1.2 Document conventions

- JSON is defined by [RFC 8259] / [RFC 7493]; member order is not semantically significant.
- Timestamps are ISO 8601 / RFC 3339 strings [RFC 3339].
- Embedded binary content SHOULD use base64url encoding.
- Parsers MUST normalize `kind` to lower case at ingress (§3.1).

### 1.3 Relationship to GeoJSON

GPJSON reuses the GeoJSON top-level `Feature` / `FeatureCollection` structures; extends `geometry` with eight additional types; and adds the core members `datasets`, `kind`, `crs` and `gpjson`. A GeoJSON-only implementation SHOULD skip extended geometries (while retaining `properties`) and ignore `datasets`, rather than raise an error.

| Consumer capability | GPJSON Feature behaviour |
|---|---|
| GeoJSON only | Reads standard `geometry` + `properties`; skips extended geometry and `datasets` |
| GeoJSON + `datasets` | Reads payloads; skips extended geometry |
| Full GPJSON 4.1 | Uses all fields |

### 1.4 Design goals

1. Express what GeoJSON cannot conveniently carry (multi-channel, streaming, N-D and 3-D volumetric data).
2. Carry tables, matrices, binaries, external references and streaming segments uniformly through `datasets`.
3. Make data self-describing, routable and auditable in a node-based pipeline via `kind` and optional profiles.
4. Keep the small core stable; place optional semantics in separate, namespaced profiles.
5. Coexist and inter-operate with established OGC models and encodings (§11).

---

## 2. Object model

### 2.1 Feature

A GPJSON Feature is a JSON object.

| Member | Requirement | Description |
|---|---|---|
| `type` | **MUST** | Fixed string `"Feature"` |
| `properties` | **MUST** | JSON object; business-attribute container (§5) |
| `gpjson` | SHOULD | Protocol version string; `"4.1"` for this specification |
| `id` | SHOULD | Unique feature identifier, string or number |
| `geometry` | SHOULD | Geometry object (§2.3 / §2.4) |
| `crs` | MAY | Coordinate reference system (§4) |
| `datasets` | MAY | Array of payload carriers (§6) |
| `dataset` | MAY | Singular shorthand for a one-element `datasets`; parsers SHOULD normalize it to `datasets: [dataset]` |

Minimal skeleton:

```json
{
  "gpjson": "4.1",
  "type": "Feature",
  "id": "evt_demo_0001",
  "crs": "local:demo_grid",
  "geometry": { "type": "Point3D", "coordinates": [120.0, 80.0, 250.0] },
  "properties": {
    "kind": "seismic.event",
    "timestamp": "2026-05-01T08:00:00Z",
    "source": "detection_node"
  },
  "datasets": []
}
```

### 2.2 FeatureCollection

| Member | Requirement | Description |
|---|---|---|
| `type` | **MUST** | `"FeatureCollection"` |
| `features` | **MUST** | Array of Feature objects |
| `gpjson` | SHOULD | Protocol version |
| `properties` | SHOULD | Collection-level metadata |
| `bbox` | SHOULD | Bounding box `[xmin, ymin, (zmin,) xmax, ymax, (zmax)]` for spatial indexing |
| `timeRange` | SHOULD | `{"start": "...", "end": "..."}` for temporal indexing |

### 2.3 Standard GeoJSON geometries

The seven standard GeoJSON geometries are fully supported: `Point`, `LineString`, `Polygon`, `MultiPoint`, `MultiLineString`, `MultiPolygon`, `GeometryCollection`, with 2-D or optional 3-D `[x, y, z]` coordinates.

### 2.4 Extended geometries

Eight additional types coexist at the same level as standard geometries, selected by `geometry.type`.

| Type | Purpose | Key members |
|---|---|---|
| `Point3D` | Explicit 3-D point | `coordinates: [x,y,z]` |
| `LineString3D` | 3-D polyline (borehole track, tunnel centre-line) | `coordinates: [[x,y,z], ...]` |
| `Polygon3D` | 3-D closed surface; outer ring + optional inner rings | `coordinates` (GeoJSON nesting) |
| `PointCloud3D` | Unstructured 3-D point set (fault picks, LiDAR) | `points`; optional `scalars` (same length) |
| `Mesh3D` | Triangulated surface (horizon, fault, free-form surface) | `vertices` (MUST), `faces` (MUST), optional `scalars`, `encoding` |
| `Grid2D` | Regular 2-D raster (DEM, planar grid) | `origin`, `spacing`, `shape`, `axes`, `values` |
| `Grid3D` | Regular 3-D voxel volume (inversion / attribute / seismic volume) | `origin`, `spacing`, `shape`, `axes`, `values` |
| `NDArray` | Arbitrary N-D array (multi-band rasters, time stacks) | `shape` (MUST), `axes`, `dtype`, `values` (MUST) |

`Grid2D` / `Grid3D` MAY be treated as common aliases of `NDArray`. Examples:

```json
{ "type": "Point3D", "coordinates": [120.0, 80.0, 250.0] }
```

```json
{
  "type": "PointCloud3D",
  "points": [[120, 80, 250], [125, 82, 245], [130, 85, 240]],
  "scalars": [0.5, 0.8, 1.2]
}
```

```json
{
  "type": "Mesh3D",
  "vertices": [[0,0,0],[1,0,0],[1,1,0],[0,1,0]],
  "faces": [[0,1,2],[0,2,3]],
  "scalars": [0.5,0.8,1.2,0.9],
  "encoding": { "compression": "none" }
}
```

```json
{
  "type": "Grid2D",
  "origin": [0.0, 0.0], "spacing": [10.0, 10.0], "shape": [200, 150],
  "axes": ["x_m", "y_m"],
  "values": { "ref": "external", "uri": "cache/dem.bin", "dtype": "float32_le" }
}
```

```json
{
  "type": "NDArray",
  "shape": [12, 4, 200, 150],
  "axes": ["time", "band", "y", "x"],
  "dtype": "float32",
  "values": { "ref": "external", "uri": "cache/field.zarr", "format": "zarr" },
  "encoding": { "compression": "zstd", "chunking": [1, 1, 200, 150] }
}
```

### 2.5 Interoperability

Even when a Feature uses an extended geometry, its `properties` and `id` remain readable and indexable by any GeoJSON tool. This is the GPJSON/GeoJSON coexistence contract.

---

## 3. The `kind` namespace

`properties.kind` (string) states what business-semantic object a Feature is; it is the key field for pipeline routing and parser dispatch.

### 3.1 Grammar

- Lower-case letters, digits and underscores only; no upper case, spaces, or hyphens.
- Namespaces are dot-separated, at most five segments.
- Parsers MUST apply `to_lower` normalization at ingress.

```abnf
kind            ::= simple-kind | namespaced-kind
simple-kind     ::= [a-z][a-z0-9_]*
namespaced-kind ::= namespace ("." segment)+
namespace       ::= [a-z][a-z0-9_]*
segment         ::= [a-z][a-z0-9_]*
```

Valid: `borehole`, `tunnel`, `seismic.event`, `com.example.signal_v2`.
Invalid: `Borehole` (upper case), `seismic-event` (hyphen), `seismic event` (space).

### 3.2 Reserved and third-party prefixes

The `gpjson.*` prefix is reserved by this specification. Standardized domain prefixes include `borehole.*`, `tunnel.*`, `fault.*`, `horizon.*`, `terrain.*`, `inversion.*`, `sensor.*`, and `seismic.*`. Third parties SHOULD use reverse-domain-name prefixes, e.g. `com.example.module.kind`.

### 3.3 Kinds used by the companion code

The reference package registers a per-domain policy by the **first** `kind` segment.

| Domain | Registered prefix | Example `kind` | Typical geometry | Payload role |
|---|---|---|---|---|
| Microseismic | `seismic.*` | `seismic.trace`, `seismic.event` | `LineString3D` (traces), `Point3D` (event) | waveform `NDArray`, event attributes |
| InSAR | `insar.*` | `insar.deformation` | `Grid2D` / `NDArray` (LOS deformation) | `los_displacement` (`NDArray`) |

> The `insar.*` namespace is introduced as a **domain extension** by this companion package. It is not yet part of the core kind registry; it follows the extension rules of §12 (additive, namespaced, ignored by older consumers) and MAY be registered formally or moved under a reverse-domain prefix without affecting the orchestration mechanism.

---

## 4. Coordinate reference systems (`crs`)

`crs` (string, OPTIONAL) declares the CRS used by the geometry and by coordinates inside `properties`.

- If absent, it SHOULD be treated as `"EPSG:4326"` (WGS84 longitude/latitude, consistent with the GeoJSON default).
- Engineering and mining use SHOULD use an explicit projected EPSG code (e.g. `"EPSG:32650"` for WGS84 / UTM zone 50N).
- A local engineering grid MAY use a custom string with the `local:` prefix, e.g. `"local:demo_grid"`, resolved by the consumer from context.
- `crs` MAY also appear at the FeatureCollection top level as a default, overridden by a Feature-level `crs`.

GPJSON deliberately departs from the RFC 7946 policy of forbidding a `crs` member, because regional projected coordinates are pervasive in geoscience and engineering; forcing WGS84 would introduce metre-to-decametre positional errors. GPJSON therefore treats CRS as a first-class field.

---

## 5. `properties` and the core `quality` block

`properties` is a JSON object holding all unstructured business attributes.

### 5.1 Common fields

| Field | Type | Requirement | Description |
|---|---|---|---|
| `kind` | string | MUST | Business-semantic classification (§3) |
| `name` | string | OPTIONAL | Human-readable name |
| `timestamp` | ISO 8601 string | SHOULD | Data production / effect time |
| `source` | string | SHOULD | Producing node/system identifier |
| `units` | object | OPTIONAL | Global unit declarations, e.g. `{"length":"m","pressure":"MPa"}` |
| `schema` | string | OPTIONAL | JSON Schema reference for compatibility checks |
| `note` | string | OPTIONAL | Free text; not used for routing |

### 5.2 `quality` — intrinsic data quality (core)

`properties.quality` describes the intrinsic quality of the data itself. In GPJSON 4.1 this block is part of the **core** (it was promoted from a draft profile so that it appears consistently across domains).

| Field | Type | Meaning |
|---|---|---|
| `completeness` | number ∈ [0,1] | Completeness (1 − missing/total) |
| `outlierRate` | number ∈ [0,1] | Fraction of flagged outliers |
| `confidence` | number ∈ [0,1] | Confidence (used for inferred results) |
| `validity` | enum | `valid` / `suspect` / `invalid` |
| `uncertainty` | object | Uncertainty structure (Gaussian / interval / quantile / ellipsoid / RMS …) |

```json
"quality": {
  "completeness": 0.92,
  "outlierRate": 0.01,
  "confidence": 0.85,
  "validity": "valid",
  "uncertainty": { "type": "gaussian", "sigma": 0.008, "unit": "m" }
}
```

The reference implementation uses the following modality-independent gates (single source of truth: `gpjson41/orch_profile.py`):

- Confidence gate `CONF_LOW = 0.60`; completeness floor `COMPLETENESS_MIN = 0.90`; outlier ceiling `OUTLIER_HIGH = 0.20`.
- Validity is derived from confidence: `confidence ≥ 0.60 → valid`; `0.35 ≤ confidence < 0.60 → suspect`; otherwise `invalid`.
- **Core degradation** is decided from the core quality block alone:
  `confidence < 0.60` **OR** `completeness < 0.90` **OR** `outlierRate > 0.20`.

This core block is the modality-independent part of the unified *state* used by the Orchestration Profile (§10).

---

## 6. `datasets` — payload carriers

### 6.1 Overview

`datasets` is an array on a Feature; each element is a dataset object carrying one payload. A Feature MAY carry multiple datasets to express a composite entity (e.g. a borehole = trajectory geometry + stratigraphy Sheet + log NDArray + core-image Binary).

Each dataset MUST contain `type`; it SHOULD contain `role` (its semantic part in a multi-dataset Feature).

| `type` | Description |
|---|---|
| `Sheet` | Static, logically complete table (`dimensions` + `source` rows) |
| `NDArray` | N-D numeric array (`shape`, `axes`, `dtype`, `values`) |
| `Matrix` | 2-D alias of `NDArray` |
| `Binary` | Embedded binary files (`files[]`, base64url, `sha256`) |
| `URI` | External resource references (`items[]` with `uri`, `format`, `sha256`, optional `auth` hint) |
| `Sequence` | A segment of a streaming/incremental feed (same shape as `Sheet` plus `stream`) |

The singular `dataset` SHOULD still be accepted as `datasets: [dataset]`; producers are RECOMMENDED to emit `datasets` directly.

### 6.2 `values` forms (NDArray / grids)

1. Inline array: `"values": [[1.0, 2.0], [3.0, 4.0]]`
2. base64url-encoded binary: `"values": "base64:eJyTYwAA..."`
3. External reference object: `"values": { "ref": "external", "uri": "...", "format": "..." }`

### 6.3 Large data and integrity

Any embedded base64 payload larger than **1 MiB** SHOULD instead use an external reference, optionally with `size` and `sha256`:

```json
"values": { "ref": "external", "uri": "cache/field.bin", "format": "raw_le",
            "size": 2400000, "sha256": "ab12cd34..." }
```

The `uri` is interpreted by the consumer context (relative path, `file://`, `https://`, `s3://`). Embedded binaries and external items SHOULD carry `sha256` for integrity, which is REQUIRED in signed/audited deployments.

### 6.4 Streaming `Sequence`

A `Sequence` uses `dimensions` + `source` like a `Sheet`, plus a SHOULD `stream` object:

```json
{
  "type": "Sequence",
  "role": "live_pressure",
  "dimensions": ["time", "value/MPa"],
  "source": [
    ["2026-05-01T10:00:00Z", 27.30],
    ["2026-05-01T10:00:01Z", 27.32]
  ],
  "stream": {
    "streamId": "sta_demo_pressure",
    "segmentSeq": 38219,
    "isComplete": false,
    "watermark": "2026-05-01T10:00:01Z"
  }
}
```

Segments sharing a `streamId` MAY be collected, validated for consistent `dimensions`, merged by time/key, and re-emitted as a static `Sheet`.

---

## 7. Time and date

GPJSON introduces no new JSON date type. All times SHOULD be ISO 8601 / RFC 3339 strings and SHOULD live under `properties`, e.g. `timestamp`, `timeStart`, `timeEnd`, `observationAt`. Fractional seconds and explicit offsets (e.g. `2026-05-01T10:00:00.123Z`, `...+08:00`) are permitted.

---

## 8. Security and integrity (summary)

- **Binary decoding:** consumers MUST verify `sha256` where present, guard against buffer overflows / parser denial-of-service, and sandbox untrusted binary content.
- **External URIs:** consumers MUST allow-list URI schemes (recommended `https://`, `s3://`, `file://`), reject dangerous schemes such as `javascript:` / `data:`, and apply authorization and path-traversal checks.
- **Signatures:** a canonical form for stable digital signatures follows [RFC 8785] (JSON Canonicalization Scheme): keys sorted by Unicode code point, shortest-round-trip numbers, Unicode NFC strings, arrays preserved in order, compact whitespace; `properties.signature` is excluded before signing. Downstream nodes that need to append lineage/processing time SHOULD emit a new signed Feature rather than mutate a signed one.
- **Embedded scripts:** an optional scripting profile permits sandboxed suggested code; implementations MAY ignore it and MUST execute any embedded script only inside a sandbox.

---

## 9. Pipeline Profile (recommended)

`properties.pipeline` carries pipeline observation, streaming, routing and governance metadata. It is strongly recommended in node-based pipeline systems.

| Field | Type | Description |
|---|---|---|
| `lineage` | array of feature ids | Direct upstream Feature ids (data provenance) |
| `derivedFrom` | array of `"id@time"` | Source feature + event time (finer than lineage) |
| `window` | object | Aggregation window `{start, end, label}` |
| `eventTime` | ISO 8601 | When the event actually occurred (data truth time) |
| `processingTime` | ISO 8601 | When the node processed the data |
| `watermark` | ISO 8601 | Stream watermark (all events ≤ this time received) |
| `latencyBudget` | number (seconds) | End-to-end latency budget |
| `qos` | object | Quality-of-service policy, e.g. `{"priority":"high","deliveryGuarantee":"at_least_once"}` |
| `sensitivity` | enum | `public` / `internal` / `restricted` |
| `scrubbed` / `scrubVersion` | bool / string | De-identification flag and algorithm version |
| `routingDecision` | enum | `local` / `cloud` / `cloud_fallback_local` |
| `modelId` / `promptVersion` | string | Model / prompt identifiers for audit |
| `health` | object | Sensor/data health indicators (free-form, **kind-scoped**) |
| `stage` | object | Life-cycle stage, e.g. `{"level":"normalized","readyForFusion":true}`; levels `raw / normalized / aggregated / fused` |
| `transport` | object | Transport metadata (e.g. MQTT topic) |

The Orchestration Profile below uses four of these locations (`stage.level`, `health`, `latencyBudget`, `qos`) and adds only one new optional member.

---

## 10. Orchestration Profile (optional extension over GPJSON 4.1)

This section is the specification of the orchestration semantics introduced in the manuscript. The reference implementation is `gpjson41/orch_profile.py` (modality-agnostic core) together with `gpjson41/policies_microseismic.py` and `gpjson41/policies_insar.py`.

### 10.1 Status, and relation to the withdrawn draft "State & Intent Profile"

- GPJSON **4.1 is not revised or version-bumped** by this profile. The Orchestration Profile is an **optional** set of members under `properties`.
- Per §12.2, *adding a new `properties` member is forward-compatible: older consumers ignore unknown members*. A 4.1-only consumer that does not know `orch_intent` simply ignores it; the object remains a valid GPJSON 4.1 Feature. The JSON Schema therefore keeps `"gpjson": {"pattern": "^4\\."}`.
- This profile **does not revive** the "State & Intent Profile" that was present in the 4.0 draft and **withdrawn in 4.1**. In 4.1 the common data-quality state already lives in the core `quality` block (§5.2), and streaming/routing state lives in the Pipeline Profile (§9). The Orchestration Profile adds only the missing **goal / expected-outcome** semantics needed for runtime, state-driven composition; it does not redefine quality.

### 10.2 Two-layer design: unified orchestration semantics, heterogeneous payloads

The key answer to cross-modality heterogeneity is a strict layering:

- **Unified orchestration layer (fixed vocabulary).** Every modality is projected onto the same small *state* and *intent* dictionary: the core `quality` block, `pipeline.stage`, `pipeline.health`, and `orch_intent` (`goal`, `outcomes`, `constraints`). The planner and execution engine operate only on this fixed dictionary.
- **Heterogeneous payload layer (unchanged).** The actual observations remain modality-specific and stay in `datasets` / extended geometry — waveforms for microseismics, interferometric deformation fields for InSAR. The profile does not normalize payloads into one array shape, nor does it change processing physics.

Domain differences are confined to (a) modality-specific *health indicators* inside `pipeline.health` and (b) per-domain reference policies selected by `kind`, never to the orchestration schema.

### 10.3 Common state projection

A 4.1 Feature is projected to the common state vector as follows (`state_view`):

| Common state | GPJSON 4.1 location |
|---|---|
| `state.quality.{completeness, outlierRate, confidence, validity}` | `properties.quality` (core) |
| `state.uncertainty` | `properties.quality.uncertainty` |
| `state.stage` | `properties.pipeline.stage.level` |
| `state.health` | `properties.pipeline.health` (free-form, kind-scoped) |

### 10.4 Common intent projection

The intent is carried by the single new optional member `properties.orch_intent` and projected (`intent_view`) as:

| Member | Type | Meaning |
|---|---|---|
| `goal` | string | The operational goal of this data object / task |
| `outcomes` | array of string | Measurable expected outcomes / acceptance signals |
| `constraints` | object | Hard/soft constraints; `pipeline.latencyBudget` is exposed as `constraints.latencyBudget_s` and `pipeline.qos` as `constraints.qos` |

```json
"orch_intent": {
  "goal": "anomaly_confirmation",
  "outcomes": ["deformation_rate_map", "anomaly_alert"],
  "constraints": { "latencyBudget_s": 86400, "accuracyClass": "standard" }
}
```

High priority is read from `constraints.qos.priority == "high"`; goal matching is by substring token (e.g. `near_boundary`, `anomaly_confirmation`, `twin_assimilation`).

### 10.5 Fixed schema, scenario-dependent values

The **schema is fixed** across applications; only the **values** of `goal` / `outcomes` / health indicators change with the scenario. This directly addresses whether intent "changes" across scenarios: the *dictionary and its location do not change*; the *goal token and its expected outcomes do*. The companion package demonstrates three intents over the same schema:

| Scenario | `goal` token | Chain behaviour |
|---|---|---|
| Trend mapping | `trend_mapping` | Short backbone chain; no confirmation / rate-up |
| Anomaly confirmation | `anomaly_confirmation` | Adds persistence + physics cross-validation; at high priority, denser acquisition |
| Digital-twin assimilation | `twin_assimilation` | Adds persistence + physics cross-validation; no acquisition rate-up |

### 10.6 Kind-scoped health and per-domain reference policies

A domain policy is registered by the first `kind` segment (`register_domain(prefix, policy)`). Each policy (i) maps modality-specific ingest estimates **deterministically** onto the core `quality` block, and (ii) returns an auditable list of conditional actions (`reference_actions`) from an explicit `ACTION_UNIVERSE`. The planner composes the node chain from this vocabulary; the unconditional processing backbone is always implied.

| Aspect | Microseismic domain | InSAR domain |
|---|---|---|
| Kind prefix | `seismic.*` | `insar.*` |
| Health indicator | `health.estSnrDb` (dB) | `health.meanCoherence` (γ̄) |
| Quality mapping | SNR + channel completeness → `confidence`; flagged/total → `outlierRate` | coherence + coverage → `confidence`; `completeness = coverage`, `outlierRate = 1 − coverage` |
| Adaptive actions | `gap_repair`, `robust_denoise`, `cross_validate`, `event_rate_up` | `gap_repair_insar`, `spatial_filter`, `temporal_persist_check`, `mogi_cross_validate`, `investigation_rate_up` |
| Physics-based confirmation | `cross_validate` | `mogi_cross_validate` (Mogi point-pressure elastic half-space) |
| Rate-up condition | `qos.priority = high` **and** goal contains `near_boundary` | goal contains `anomaly_confirmation` **and** `qos.priority = high` |
| Confirmation/assimilation goal | — | `anomaly_confirmation` or `twin_assimilation` ⇒ persistence + Mogi cross-validation |

Reference constants used by the implementation (physical, calibrated by diagnostics — never tuned to a target score):

- Core: `CONF_LOW = 0.60`, `COMPLETENESS_MIN = 0.90`, `OUTLIER_HIGH = 0.20`.
- Microseismic: low-SNR screening gate `SNR_LOW_DB = 8 dB`.
- InSAR: pixel coherence gate `0.35`; adaptive-filter mean-coherence threshold `0.75`; confidence-sigmoid midpoint `0.55`, scale `0.12`; Cramér–Rao phase-noise model with `L = 20` looks and Sentinel-1 wavelength `λ = 56 mm`.

### 10.7 Minimal worked example (InSAR, abbreviated)

```json
{
  "gpjson": "4.1",
  "type": "Feature",
  "id": "insar_demo_frame_0007",
  "crs": "local:demo_grid",
  "geometry": { "type": "Grid2D", "origin": [0, 0], "spacing": [40, 40],
               "shape": [120, 120], "axes": ["x_m", "y_m"] },
  "properties": {
    "kind": "insar.deformation",
    "timestamp": "2026-05-01T06:00:00Z",
    "source": "insar_wrap_node",
    "quality": {
      "completeness": 0.95,
      "outlierRate": 0.05,
      "confidence": 0.93,
      "validity": "valid",
      "uncertainty": { "type": "phase_sigma_crlb", "looks": 20, "wavelength_mm": 56 }
    },
    "pipeline": {
      "stage": { "level": "normalized", "readyForFusion": true },
      "latencyBudget": 86400,
      "qos": { "priority": "high", "deliveryGuarantee": "at_least_once" },
      "health": { "meanCoherence": 0.948 }
    },
    "orch_intent": {
      "goal": "anomaly_confirmation",
      "outcomes": ["deformation_rate_map", "anomaly_alert"],
      "constraints": { "latencyBudget_s": 86400, "accuracyClass": "standard" }
    }
  },
  "datasets": [
    { "type": "NDArray", "role": "los_displacement", "shape": [120, 120],
      "axes": ["y_m", "x_m"], "dtype": "float32",
      "values": { "ref": "external", "uri": "payload/insar_frame_0007.npy", "format": "npy" } }
  ]
}
```

The microseismic object is structurally identical at the orchestration layer; only `kind` (`seismic.trace`), the health indicator (`estSnrDb`), the latency budget (seconds-scale), the goal (`rockburst_event_detection`) and the waveform payload differ. Full objects are in `examples/insar_example.json` and `examples/microseismic_example.json`.

### 10.8 What this profile deliberately does not do

- It does **not** specify the internal layout or processing of modality-specific payloads (waveforms, interferograms); those remain domain-specific.
- It does **not** replace domain physics, quality science, or sensor-specific uncertainty models; it only routes them through a common quality/state contract.
- It does **not** add new top-level structures or alter existing core semantics, so no version bump is triggered (§12).

---

## 11. Relationship to OGC standards

GPJSON does not compete with OGC encoding, capability or service standards; it occupies a different layer — a **message-level, data-carried, runtime orchestration contract**. OGC standards largely address static capability description, observation/coverage *encoding*, and deployment-time services, whereas GPJSON travels **with each data object** and tells a node graph how that object should be composed and routed **at runtime**, given its current state and intent. The two are designed to coexist and be inter-converted.

| OGC standard (document) | What it standardizes | Layer | How GPJSON coexists |
|---|---|---|---|
| SensorML (`12-000r2`, encoding `23-000`) | Systems, sensors and process/process-chain descriptions and capabilities | Static, design/deployment-time capability model | SensorML describes *what a node/system is*; GPJSON carries per-message runtime `quality`/`pipeline`/`orch_intent` for *what this data object needs now*. A chain described in SensorML can be selected at runtime by GPJSON intent. |
| Observations & Measurements — O&M (`07-022r1`, ISO 19156) | The observation/measurement feature model and encoding | Observation encoding | A GPJSON Feature can encode an observation result in `datasets`; O&M models the observation act/sampling, GPJSON adds the routing/quality/intent contract around it. |
| CoverageJSON (`21-069r2`) | Encoding of gridded coverages | Field/coverage encoding | GPJSON `Grid2D`/`Grid3D`/`NDArray` payloads map to Coverages; CoverageJSON encodes the field, GPJSON wraps it with orchestration metadata. |
| SensorThings API (Part 1 `15-078r6`, Part 2 `17-079r1`) | REST/STA IoT service, CRUD/query, tasking | Deployment-time service API | Service-level; GPJSON is the message exchanged between pipeline nodes and can flow over MQTT/HTTP alongside a SensorThings deployment. |
| STAC 1.1 (2025 Community Standard); Cloud-Optimized GeoTIFF (`21-026`) | Cataloguing of at-rest assets; cloud-optimized raster tiles | Discovery / at-rest assets | GPJSON `URI` datasets MAY reference STAC items or COG assets; STAC/COG handle discovery/tiling, GPJSON handles runtime composition. |

**Boundary statement.** It is acknowledged that SensorML can describe process chains and CoverageJSON can encode rasters. Their focus is static capability description, encoding and deployment-time exchange. They do not define a compact contract that is carried inline with every message and that drives runtime, state/intent-based composition across heterogeneous modalities. GPJSON supplies exactly that thin profile, and it can be translated to/from the OGC representations above rather than replacing them. OGC Testbed-13 (`17-029r1`) and Testbed-15 (`19-022r1`) experiments reflect the same direction toward event-driven, orchestrated geospatial processing.

---

## 12. Versioning and forward compatibility

GPJSON uses the `gpjson` member for the core protocol version (the companion schema enforces `"pattern": "^4\\."`).

| Change | Compatibility action |
|---|---|
| Add a `properties` member (e.g. `orch_intent`) | Forward-compatible; older consumers ignore the unknown member — **no version bump** |
| Add a dataset `role` | Forward-compatible; older consumers skip unknown roles |
| Add a `kind` | Older consumers fall back to the generic parsing path |
| Remove a core field / change core field semantics | MUST raise the minor version (4.1 → 4.2) |
| Change the top-level Feature/Dataset structure | MUST raise the major version (4.x → 5.0) |

Parsers SHOULD check `gpjson` and return an explicit error for an unsupported major version. Because the Orchestration Profile only adds the optional `orch_intent` member and reuses existing core `quality` and Pipeline Profile members, it is delivered entirely under GPJSON **4.1** without a core version change.

> **Historical note.** The 4.0 draft contained a "State & Intent Profile"; 4.1 withdrew it, promoting the common quality fields to the core and retaining pipeline metadata in the Pipeline Profile. The Orchestration Profile in §10 is a new, additive, namespaced extension and is not the withdrawn profile.

---

## 13. Reference implementation and examples

The companion repository contains a modality-independent Python reference subset:

| Path | Contents |
|---|---|
| `gpjson41/schema_feat.json` | JSON Schema (Draft-07) for the subset used by the package |
| `gpjson41/builder.py`, `gpjson41/parser.py` | Builder/reader for Features, `quality`, `pipeline`, `orch_intent`, datasets |
| `gpjson41/orch_profile.py` | Modality-agnostic state/intent projection, quality gates, policy registry |
| `gpjson41/policies_microseismic.py`, `gpjson41/policies_insar.py` | Per-domain quality mappings and auditable action policies |
| `configs/nkg/nodes_microseismic.yaml`, `nodes_insar.yaml` | Declarative node knowledge graphs |
| `examples/microseismic_example.json`, `examples/insar_example.json` | Two schema-valid worked objects |
| `tests/test_smoke.py` | Ten smoke tests (protocol round-trip/schema, detection, planner routing for both modalities, intent switching) |

Builder sketch:

```python
from gpjson41.builder import GPJSONBuilder

f = (GPJSONBuilder("insar.deformation", "insar_demo_0007", crs="local:demo_grid")
     .set_point3d([0.0, 0.0, 0.0])
     .set_source("insar_wrap_node")
     .add_quality(completeness=0.95, confidence=0.93, validity="valid")
     .set_pipeline(stage={"level": "normalized", "readyForFusion": True},
                   health={"meanCoherence": 0.948},
                   latency_budget=86400,
                   qos={"priority": "high", "deliveryGuarantee": "at_least_once"})
     .set_intent(goal="anomaly_confirmation",
                 outcomes=["deformation_rate_map", "anomaly_alert"],
                 constraints={"latencyBudget_s": 86400,
                              "accuracyClass": "standard"})
     .build())
```

> The bundled JSON Schema is a **subset** scoped to the reproducible experiments; the complete field set is as defined in this document and the full GPJSON 4.1 specification. Reproduce the experiments with `python experiments/run_all.py` and validate with `python -m tests.test_smoke`.

---

## 14. Normative and related references

- [RFC 2119] Bradner, S., *Key words for use in RFCs to Indicate Requirement Levels*.
- [RFC 3339] *Date and Time on the Internet: Timestamps*.
- [RFC 7493] *The I-JSON Message Format*.
- [RFC 7946] Butler, H. et al., *The GeoJSON Format*.
- [RFC 8259] *The JavaScript Object Notation (JSON) Data Interchange Format*.
- [RFC 8785] *JSON Canonicalization Scheme (JCS)*.
- OGC, *SensorML: System and Sensor Description*, `12-000r2` (encoding `23-000`), docs.ogc.org.
- OGC, *Observations and Measurements (O&M)*, `07-022r1`; ISO 19156.
- OGC, *CoverageJSON*, `21-069r2`.
- OGC, *SensorThings API Part 1: Sensing*, `15-078r6`; *Part 2: Tasking*, `17-079r1`.
- OGC, *Cloud Optimized GeoTIFF (COG)*, `21-026`; *STAC Specification* 1.1 (2025 Community Standard).
- OGC Testbed-13 Engineering Report, `17-029r1`; Testbed-15 Engineering Report, `19-022r1`.
- NetCDF Format Specification; Zarr v3 Specification (external array references).

---

## 15. Citation

Please cite both the companion article and the archived code. Citation metadata is maintained in `../CITATION.cff`; the Zenodo DOI and repository URL are filled in at release and reproduced here:

- Code: `https://doi.org/10.5281/zenodo.XXXXXXX` (placeholder; minted at the v1.0.0 release)
- Repository: `https://github.com/aben-lee/igdo-engine-repro`

*This specification document is licensed under CC-BY-4.0; the reference code is licensed under the BSD 3-Clause License.*
