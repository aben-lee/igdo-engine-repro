"""Builder for GPJSON 4.1 Feature / FeatureCollection objects.

Large numeric payloads are written out of band (``.npy``) and referenced with a
``{ref: external, uri, format}`` values object, mirroring the specification's
recommendation for payloads above ~1 MiB and exercising the URI/NDArray path
used by the real platform.
"""
from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Optional

import numpy as np

GPJSON_VERSION = "4.1"


def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"


class GPJSONBuilder:
    """Fluent builder for a single GPJSON 4.1 Feature."""

    def __init__(self, kind: str, feature_id: str, crs: str = "local:mine_grid"):
        self._doc: Dict[str, Any] = {
            "gpjson": GPJSON_VERSION,
            "type": "Feature",
            "id": feature_id,
            "crs": crs,
            "geometry": None,
            "properties": {"kind": kind.lower()},
            "datasets": [],
        }

    # --- geometry -------------------------------------------------------
    def set_geometry(self, geometry: Dict[str, Any]) -> "GPJSONBuilder":
        self._doc["geometry"] = geometry
        return self

    def set_point3d(self, xyz: Iterable[float]) -> "GPJSONBuilder":
        self._doc["geometry"] = {"type": "Point3D", "coordinates": [float(v) for v in xyz]}
        return self

    # --- properties -----------------------------------------------------
    def set_source(self, source: str) -> "GPJSONBuilder":
        self._doc["properties"]["source"] = source
        return self

    def set_timestamp(self, ts: Optional[str] = None) -> "GPJSONBuilder":
        self._doc["properties"]["timestamp"] = ts or _utcnow_iso()
        return self

    def add_property(self, key: str, value: Any) -> "GPJSONBuilder":
        self._doc["properties"][key] = value
        return self

    def add_quality(
        self,
        completeness: Optional[float] = None,
        outlier_rate: Optional[float] = None,
        confidence: Optional[float] = None,
        validity: Optional[str] = None,
        uncertainty: Optional[Dict[str, Any]] = None,
    ) -> "GPJSONBuilder":
        """Populate the CORE v4.1 quality block (only provided keys are set)."""
        q = self._doc["properties"].setdefault("quality", {})
        if completeness is not None:
            q["completeness"] = float(completeness)
        if outlier_rate is not None:
            q["outlierRate"] = float(outlier_rate)
        if confidence is not None:
            q["confidence"] = float(confidence)
        if validity is not None:
            q["validity"] = validity
        if uncertainty is not None:
            q["uncertainty"] = uncertainty
        return self

    def set_pipeline(
        self,
        event_time: Optional[str] = None,
        processing_time: Optional[str] = None,
        watermark: Optional[str] = None,
        latency_budget: Optional[float] = None,
        qos: Optional[Dict[str, Any]] = None,
        stage: Optional[Dict[str, Any]] = None,
        health: Optional[Dict[str, Any]] = None,
        lineage: Optional[List[str]] = None,
        transport: Optional[Dict[str, Any]] = None,
    ) -> "GPJSONBuilder":
        """Populate the recommended Pipeline Profile (appendix A of the spec)."""
        p = self._doc["properties"].setdefault("pipeline", {})
        if event_time is not None:
            p["eventTime"] = event_time
        if processing_time is not None:
            p["processingTime"] = processing_time
        if watermark is not None:
            p["watermark"] = watermark
        if latency_budget is not None:
            p["latencyBudget"] = float(latency_budget)
        if qos is not None:
            p["qos"] = qos
        if stage is not None:
            p["stage"] = stage
        if health is not None:
            p["health"] = health
        if lineage is not None:
            p["lineage"] = list(lineage)
        if transport is not None:
            p["transport"] = transport
        return self

    def set_intent(
        self,
        goal: str,
        outcomes: Optional[List[str]] = None,
        constraints: Optional[Dict[str, Any]] = None,
    ) -> "GPJSONBuilder":
        """Populate the OPTIONAL orchestration extension (paper's intent)."""
        self._doc["properties"]["orch_intent"] = {
            "goal": goal,
            "outcomes": outcomes or [],
            "constraints": constraints or {},
        }
        return self

    # --- datasets -------------------------------------------------------
    def add_dataset_ndarray(
        self,
        role: str,
        array: np.ndarray,
        axes: Optional[List[str]] = None,
        units: Optional[str] = None,
        out_dir: Optional[str] = None,
        file_stem: Optional[str] = None,
        inline_limit_bytes: int = 64 * 1024,
    ) -> "GPJSONBuilder":
        """Attach an NDArray payload; small arrays inline, large arrays out of band."""
        array = np.asarray(array, dtype=np.float32)
        ds: Dict[str, Any] = {
            "type": "NDArray",
            "role": role,
            "shape": list(array.shape),
            "dtype": "float32",
        }
        if axes is not None:
            ds["axes"] = list(axes)
        if units is not None:
            ds["units"] = units
        nbytes = array.nbytes
        if nbytes <= inline_limit_bytes:
            ds["values"] = array.tolist()
        else:
            assert out_dir is not None and file_stem is not None, (
                "out_dir and file_stem required for out-of-band payloads"
            )
            os.makedirs(out_dir, exist_ok=True)
            uri = f"cache/{file_stem}__{role}.npy"
            out_path = os.path.join(out_dir, uri)
            os.makedirs(os.path.dirname(out_path), exist_ok=True)
            np.save(out_path, array, allow_pickle=False)
            ds["values"] = {"ref": "external", "uri": uri, "format": "npy",
                            "size": int(nbytes)}
        self._doc["datasets"].append(ds)
        return self

    def add_dataset_sheet(
        self, role: str, dimensions: List[str], rows: List[List[Any]]
    ) -> "GPJSONBuilder":
        self._doc["datasets"].append(
            {"type": "Sheet", "role": role, "dimensions": dimensions,
             "source": rows}
        )
        return self

    def add_dataset_sequence(
        self,
        role: str,
        dimensions: List[str],
        rows: List[List[Any]],
        stream_id: str,
        segment_seq: int,
        is_complete: bool,
        watermark: Optional[str] = None,
    ) -> "GPJSONBuilder":
        ds: Dict[str, Any] = {
            "type": "Sequence",
            "role": role,
            "dimensions": dimensions,
            "source": rows,
            "stream": {
                "streamId": stream_id,
                "segmentSeq": int(segment_seq),
                "isComplete": bool(is_complete),
            },
        }
        if watermark is not None:
            ds["stream"]["watermark"] = watermark
        self._doc["datasets"].append(ds)
        return self

    def build(self) -> Dict[str, Any]:
        # GPJSON allows geometry-less features, but the builder makes the
        # omission explicit rather than leaving a null key around.
        if self._doc["geometry"] is None:
            self._doc.pop("geometry", None)
        return json.loads(json.dumps(self._doc))  # round-trip to fail on NaN etc.


def feature_collection(features: List[Dict[str, Any]], **props) -> Dict[str, Any]:
    return {
        "gpjson": GPJSON_VERSION,
        "type": "FeatureCollection",
        "properties": props,
        "features": features,
    }


def write_feature(path: str, feature: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(feature, fh, ensure_ascii=False, indent=2)


def write_collection(path: str, collection: Dict[str, Any]) -> None:
    write_feature(path, collection)
