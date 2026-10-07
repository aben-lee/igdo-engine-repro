"""GPJSON 4.1 protocol layer for the IGDO-Engine reproducibility package.

The wire objects produced and consumed here follow the *released* GPJSON 4.1
specification (GeoJSON-compatible Feature/FeatureCollection, ``datasets``
payloads, the core ``quality`` block and the recommended Pipeline Profile).

The paper's lifecycle semantics (state / intent) are implemented as a THIN
optional orchestration extension on top of 4.1 (``orch_profile``):

    state.quality      -> properties.quality
    state.uncertainty  -> properties.quality.uncertainty
    state.stage        -> properties.pipeline.stage.level
    state.health       -> properties.pipeline.health
    intent.latency     -> properties.pipeline.latencyBudget / qos
    intent.goal        -> properties.orch_intent.goal   (optional, ignorable)

A consumer that only understands GPJSON 4.1 can ignore ``orch_intent`` and
still use every other field, which is exactly the forward-compatibility
behaviour required by the specification.
"""
from .builder import GPJSONBuilder, write_feature, write_collection
from .parser import GPJSONParser
from . import orch_profile
from . import policies_microseismic  # registers the 'seismic' domain policy
from . import policies_insar         # registers the 'insar' domain policy

__all__ = [
    "GPJSONBuilder",
    "GPJSONParser",
    "orch_profile",
    "policies_microseismic",
    "policies_insar",
    "write_feature",
    "write_collection",
]
