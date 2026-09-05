import sys
from pathlib import Path

# Ensure the app package is discoverable when this package is imported
# directly (the `backend` package must be on sys.path for `app` to resolve).
backend_dir = str(Path(__file__).resolve().parents[2])
root_dir = str(Path(__file__).resolve().parents[3])
for p in (backend_dir, root_dir):
    if p not in sys.path:
        sys.path.insert(0, p)

from app.geo.distance import (
    haversine_distance,
    haversine_km,
    calculate_distance,
    find_nearest_distance,
    calculate_target_distances,
)
from app.geo.spatial_context import (
    compute_geospatial_context,
    analyze_category_distances,
)

__all__ = [
    "haversine_distance",
    "haversine_km",
    "calculate_distance",
    "find_nearest_distance",
    "calculate_target_distances",
    "compute_geospatial_context",
    "analyze_category_distances",
]