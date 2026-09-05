"""
Schemas package for THERMOSCOPE-AI MVP.
"""
import sys
from pathlib import Path

# Ensure the app package is discoverable when this module is executed directly.
backend_dir = str(Path(__file__).resolve().parents[2])
root_dir = str(Path(__file__).resolve().parents[3])
for p in (backend_dir, root_dir):
    if p not in sys.path:
        sys.path.insert(0, p)

from app.schemas.hotspot import Hotspot, HotspotBase, normalize_confidence, row_to_hotspot
from app.schemas.spatial_context import SpatialContext
from app.schemas.analysis import HotspotAnalysis

__all__ = [
    "Hotspot",
    "HotspotBase",
    "normalize_confidence",
    "row_to_hotspot",
    "SpatialContext",
    "HotspotAnalysis",
]
