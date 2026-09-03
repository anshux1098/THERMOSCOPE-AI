"""
Geo package for spatial calculations and distance algorithms.
"""
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
