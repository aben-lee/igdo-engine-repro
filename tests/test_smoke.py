"""Smoke tests: protocol round-trip, detection, and planner routing.

Run from the project root:  python -m tests.test_smoke
"""
from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import jsonschema  # noqa: E402

from gpjson41 import GPJSONBuilder, GPJSONParser, write_feature  # noqa: E402
from gpjson41.orch_profile import state_view, intent_view  # noqa: E402
from synth import MSConfig, generate_window, make_station_network  # noqa: E402
from synth.microseismic import reference_signal_rms  # noqa: E402
from engine.nkg import NodeKnowledgeGraph  # noqa: E402
from engine.planner import make_plan  # noqa: E402
from engine.runners import run_igdo, run_static  # noqa: E402
from synth.insar import InSARConfig, generate_scene as gen_insar  # noqa: E402
from engine.runners_insar import (run_igdo as run_igdo_insar,  # noqa: E402
                                  run_static as run_static_insar)

NKG = NodeKnowledgeGraph.from_yaml(
    str(ROOT / "configs" / "nkg" / "nodes_microseismic.yaml"))
NKG_INSAR = NodeKnowledgeGraph.from_yaml(
    str(ROOT / "configs" / "nkg" / "nodes_insar.yaml"))
SCHEMA = json.loads((ROOT / "gpjson41" / "schema_feat.json").read_text(
    encoding="utf-8"))


class TestProtocol(unittest.TestCase):
    def _feature(self, tmp: str):
        cfg = MSConfig()
        stations = make_station_network(cfg, seed=0)
        ref = reference_signal_rms(cfg, stations)
        rng = np.random.default_rng(1)
        feat, _ = generate_window(cfg, stations, 20.0, 1.0, True, rng, ref)
        return feat

    def test_inline_roundtrip_and_schema(self):
        feat = self._feature(None)
        jsonschema.validate(feat, SCHEMA)
        p = GPJSONParser(feat)
        self.assertEqual(p.kind, "seismic.trace")
        wf = p.load_role_array("waveform")
        self.assertEqual(wf.shape, (8, 5000))
        self.assertAlmostEqual(p.completeness, 1.0, places=6)
        self.assertEqual(p.stage, "raw")
        self.assertIn("rockburst", p.goal)

    def test_external_npy_roundtrip(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg = MSConfig()
            stations = make_station_network(cfg, seed=0)
            ref = reference_signal_rms(cfg, stations)
            rng = np.random.default_rng(2)
            feat, _ = generate_window(cfg, stations, 10.0, 1.0, False, rng, ref)
            # rebuild with external payload under tmp
            wf = GPJSONParser(feat).load_role_array("waveform")
            b = GPJSONBuilder("seismic.trace", "ext_test")
            b.set_point3d(np.zeros(3))
            b.add_dataset_ndarray("waveform", wf, out_dir=tmp,
                                  file_stem="ext_test")
            feat2 = b.build()
            write_feature(str(Path(tmp) / "f.json"), feat2)
            p = GPJSONParser(feat2, base_dir=tmp)
            wf2 = p.load_role_array("waveform")
            self.assertTrue(np.allclose(wf, wf2))


class TestDetection(unittest.TestCase):
    def test_high_snr_event_detected_both(self):
        cfg = MSConfig()
        stations = make_station_network(cfg, seed=0)
        ref = reference_signal_rms(cfg, stations)
        n_det_s, n_det_i = 0, 0
        for w in range(5):
            rng = np.random.default_rng(100 + w)
            feat, _ = generate_window(cfg, stations, 20.0, 1.0, True, rng, ref)
            n_det_s += run_static(feat, NKG)["n_alerts"]
            n_det_i += run_igdo(feat, NKG)["n_alerts"]
        self.assertGreaterEqual(n_det_s, 4)
        self.assertGreaterEqual(n_det_i, 4)


class TestPlanner(unittest.TestCase):
    def _plan(self, snr, comp):
        state = {"quality": {"completeness": comp, "outlierRate": max(0.0, 1 - comp),
                             "confidence": 0.95 if snr >= 10 else 0.3,
                             "validity": "valid"},
                 "health": {"estSnrDb": snr}, "stage": "raw"}
        intent = {"goal": "rockburst_event_detection", "constraints": {},
                  "outcomes": []}
        return make_plan(NKG, state, intent)["nodes"]

    def test_low_quality_triggers_adaptive_nodes(self):
        nodes = self._plan(0.0, 0.70)
        self.assertIn("gap_repair", nodes)
        self.assertIn("robust_denoise", nodes)
        self.assertIn("cross_validate", nodes)
        self.assertNotIn("bandpass_wide", nodes)  # replaced by robust_denoise

    def test_high_quality_uses_backbone(self):
        nodes = self._plan(20.0, 1.0)
        self.assertNotIn("gap_repair", nodes)
        self.assertNotIn("robust_denoise", nodes)
        self.assertNotIn("cross_validate", nodes)
        self.assertIn("bandpass_wide", nodes)

    def test_rate_up_on_near_boundary_high_priority(self):
        state = {"quality": {"completeness": 1.0, "outlierRate": 0.0,
                             "confidence": 0.95, "validity": "valid"},
                 "health": {"estSnrDb": 20.0}, "stage": "raw"}
        intent = {"goal": "rockburst_event_detection_near_boundary",
                  "constraints": {"qos": {"priority": "high"}}, "outcomes": []}
        nodes = make_plan(NKG, state, intent)["nodes"]
        self.assertIn("event_rate_up", nodes)


class TestInSAR(unittest.TestCase):
    """Cross-modality (InSAR) instance on the SAME engine/protocol."""

    @classmethod
    def setUpClass(cls):
        cls.cfg = InSARConfig()

    def _scene(self, mc, has_source, seed, goal="anomaly_confirmation",
               prio="normal", budget=86400.0):
        return gen_insar(self.cfg, mc, has_source,
                         np.random.default_rng(seed), 0, goal=goal,
                         priority=prio, latency_budget_s=budget)

    def test_insar_schema_and_roundtrip(self):
        feat, _ = self._scene(0.85, True, 20261005)
        jsonschema.validate(feat, SCHEMA)
        p = GPJSONParser(feat)
        self.assertEqual(p.kind, "insar.deformation")
        los = p.load_role_array("los_displacement")
        self.assertEqual(los.shape, (16, 64, 64))
        self.assertEqual(feat["geometry"]["type"], "Grid2D")
        self.assertIn("orch_intent", feat["properties"])

    def test_high_coherence_source_confirmed_by_igdo(self):
        feat, truth = self._scene(0.85, True, 20261005)
        rs = run_static_insar(feat, NKG_INSAR)
        ri = run_igdo_insar(feat, NKG_INSAR)
        self.assertGreaterEqual(rs["n_alerts"], 1)          # backbone detects
        confirmed = [a for a in ri["alerts"] if a["confirmed"]]
        self.assertTrue(confirmed)                          # Mogi confirmation
        a = min(ri["alerts"],
                key=lambda z: np.hypot(z["loc_x"] - truth["source_x"],
                                       z["loc_y"] - truth["source_y"]))
        self.assertLessEqual(np.hypot(a["loc_x"] - truth["source_x"],
                                      a["loc_y"] - truth["source_y"]), 120.0)
        self.assertIsNotNone(a["depth_m"])
        self.assertLess(a["nrss"], 0.4)

    def test_low_coherence_igdo_zero_false_alarm_static_alarms(self):
        feat, _ = self._scene(0.45, False, 20261005 + 5000)
        ri = run_igdo_insar(feat, NKG_INSAR)
        rs = run_static_insar(feat, NKG_INSAR)
        self.assertEqual(ri["n_alerts"], 0)     # persistence + Mogi reject noise
        self.assertGreaterEqual(rs["n_alerts"], 1)

    def test_intent_drives_the_chain_schema_unchanged(self):
        f_map, _ = self._scene(0.85, True, 777,
                               goal="deformation_trend_mapping", prio="normal")
        f_cfm, _ = self._scene(0.85, True, 777,
                               goal="anomaly_confirmation", prio="high",
                               budget=3600.0)
        jsonschema.validate(f_map, SCHEMA)
        jsonschema.validate(f_cfm, SCHEMA)
        plan_map = run_igdo_insar(f_map, NKG_INSAR)["plan"]
        plan_cfm = run_igdo_insar(f_cfm, NKG_INSAR)["plan"]
        self.assertNotIn("temporal_persist_check", plan_map)
        self.assertNotIn("mogi_inversion", plan_map)
        self.assertNotIn("investigation_rate_up", plan_map)
        self.assertIn("investigation_rate_up", plan_cfm)
        self.assertIn("temporal_persist_check", plan_cfm)
        self.assertIn("mogi_cross_validate", plan_cfm)
        self.assertIn("mogi_inversion", plan_cfm)


if __name__ == "__main__":
    unittest.main(verbosity=2)
