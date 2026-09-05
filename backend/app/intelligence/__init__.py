"""
Intelligence package for THERMOSCOPE-AI.
Includes weak supervision labeling functions, rules, ML models, and hybrid consensus engines.
"""
import sys
from pathlib import Path

# Ensure the app package is discoverable when this module is executed directly.
backend_dir = str(Path(__file__).resolve().parents[2])
root_dir = str(Path(__file__).resolve().parents[3])
for p in (backend_dir, root_dir):
    if p not in sys.path:
        sys.path.insert(0, p)

from app.intelligence.labeling_functions import (
    ABSTAIN,
    INDUSTRIAL_FIRE,
    GAS_FLARE,
    MINING_ACTIVITY,
    AGRICULTURAL_BURN,
    FOREST_NATURAL_FIRE,
    INDUSTRIAL_PROCESS_HEAT,
    UNCLASSIFIED,
    ALL_LABELING_FUNCTIONS,
    LABELING_FUNCTION_MAP,
    apply_labeling_functions,
    aggregate_votes,
    compute_vote_summary,
    get_feature,
    get_numeric_feature,
    get_distance_meters,
    get_frp,
    get_brightness,
    get_confidence,
    is_night,
    get_firms_type,
    is_missing,
)

__all__ = [
    "ABSTAIN",
    "INDUSTRIAL_FIRE",
    "GAS_FLARE",
    "MINING_ACTIVITY",
    "AGRICULTURAL_BURN",
    "FOREST_NATURAL_FIRE",
    "INDUSTRIAL_PROCESS_HEAT",
    "UNCLASSIFIED",
    "ALL_LABELING_FUNCTIONS",
    "LABELING_FUNCTION_MAP",
    "apply_labeling_functions",
    "aggregate_votes",
    "compute_vote_summary",
    "get_feature",
    "get_numeric_feature",
    "get_distance_meters",
    "get_frp",
    "get_brightness",
    "get_confidence",
    "is_night",
    "get_firms_type",
    "is_missing",
]
