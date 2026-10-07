"""Minimal reader for GPJSON 4.1 objects used by the repro package."""
from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional

import numpy as np


class GPJSONParser:
    def __init__(self, doc: Dict[str, Any], base_dir: Optional[str] = None):
        self.doc = doc
        self.base_dir = base_dir  # for resolving out-of-band payload URIs

    @classmethod
    def from_file(cls, path: str) -> "GPJSONParser":
        with open(path, "r", encoding="utf-8") as fh:
            return cls(json.load(fh), base_dir=os.path.dirname(os.path.abspath(path)))

    # --- identity -------------------------------------------------------
    @property
    def kind(self) -> str:
        return str(self.doc.get("properties", {}).get("kind", "")).lower()

    @property
    def feature_id(self) -> str:
        return str(self.doc.get("id"))

    @property
    def crs(self) -> str:
        return self.doc.get("crs", "EPSG:4326")

    def get_property(self, key: str, default: Any = None) -> Any:
        return self.doc.get("properties", {}).get(key, default)

    # --- v4.1 core quality ---------------------------------------------
    @property
    def quality(self) -> Dict[str, Any]:
        return self.get_property("quality", {}) or {}

    @property
    def completeness(self) -> float:
        return float(self.quality.get("completeness", 1.0))

    @property
    def outlier_rate(self) -> float:
        return float(self.quality.get("outlierRate", 0.0))

    @property
    def confidence(self) -> float:
        return float(self.quality.get("confidence", 1.0))

    @property
    def validity(self) -> str:
        return str(self.quality.get("validity", "valid"))

    @property
    def uncertainty(self) -> Dict[str, Any]:
        return self.quality.get("uncertainty", {}) or {}

    # --- pipeline profile ----------------------------------------------
    @property
    def pipeline(self) -> Dict[str, Any]:
        return self.get_property("pipeline", {}) or {}

    @property
    def stage(self) -> str:
        return str(self.pipeline.get("stage", {}).get("level", "raw"))

    @property
    def health(self) -> Dict[str, Any]:
        return self.pipeline.get("health", {}) or {}

    @property
    def latency_budget(self) -> Optional[float]:
        v = self.pipeline.get("latencyBudget")
        return None if v is None else float(v)

    @property
    def qos(self) -> Dict[str, Any]:
        return self.pipeline.get("qos", {}) or {}

    # --- optional orchestration intent ---------------------------------
    @property
    def intent(self) -> Dict[str, Any]:
        return self.get_property("orch_intent", {}) or {}

    @property
    def goal(self) -> str:
        return str(self.intent.get("goal", ""))

    # --- datasets -------------------------------------------------------
    def datasets(self) -> List[Dict[str, Any]]:
        return self.doc.get("datasets", []) or []

    def dataset_by_role(self, role: str) -> Optional[Dict[str, Any]]:
        for ds in self.datasets():
            if ds.get("role") == role:
                return ds
        return None

    def load_array(self, dataset: Dict[str, Any]) -> np.ndarray:
        """Resolve an NDArray/Matrix payload (inline list or out-of-band npy)."""
        values = dataset["values"]
        if isinstance(values, list):
            return np.asarray(values, dtype=np.float32)
        if isinstance(values, dict) and values.get("ref") == "external":
            uri = values["uri"]
            assert self.base_dir is not None, "base_dir required for URI payloads"
            path = os.path.normpath(os.path.join(self.base_dir, uri))
            return np.load(path, allow_pickle=False).astype(np.float32)
        raise ValueError(f"Unsupported values form: {type(values)}")

    def load_role_array(self, role: str) -> Optional[np.ndarray]:
        ds = self.dataset_by_role(role)
        return None if ds is None else self.load_array(ds)
