"""
constants.py
Canonical constants and single source of truth for THERMOSCOPE-AI (SIH26162).

Architecture:
- Pure configuration and taxonomy definitions.
- Zero external runtime dependencies or business logic.
- Serves as the central reference for:
  * 7-class hotspot classification taxonomy & display mappings
  * OSM granular site types vs 7 canonical geographic categories
  * Canonical spatial distance keys & schema field names
  * NASA FIRMS datasets, confidence levels, and type codes
  * Spatial query and proximity defaults
"""
from typing import Dict, List

# ---------------------------------------------------------------------------
# 1. OSM Granular Site Types
# Returned by granular OSM classification (e.g. classify_osm_site() in osm_service.py)
# ---------------------------------------------------------------------------
SITE_TYPES: List[str] = [
    "refinery",              # Petroleum / chemical refinery
    "oil_gas",               # Oil & gas extraction / processing
    "mining",                # Quarry / coal mine / mineral extraction
    "power_plant",           # Power generation station
    "factory",               # General factory / manufacturing works
    "industrial_zone",       # Broad industrial land-use zone
    "power_infrastructure",  # Grid substation / power infrastructure
    "volcano",               # Natural volcanic thermal source
    "other_industrial",      # Uncategorized industrial tag
]

# ---------------------------------------------------------------------------
# 2. Canonical OSM Geographic Categories (7 Target Buckets)
# Used by the spatial distance pipeline (osm_service.py, distance.py, spatial_context.py)
# Note: OSM_CATEGORIES represents high-level spatial buckets, distinct from granular SITE_TYPES.
# ---------------------------------------------------------------------------
OSM_CATEGORIES: List[str] = [
    "industry",          # General factories / manufacturing / industrial zones
    "refinery",          # Petroleum / chemical refineries
    "oil_gas",           # Oil & gas extraction / processing facilities
    "mining",            # Mining sites / quarries / coal resource areas
    "forest",            # Forest / woodlands / scrub
    "agriculture",       # Farmland / crops / orchards
    "power_plant",       # Power generation plants
]

# Canonical summary distance dictionary keys (output of calculate_target_distances)
DISTANCE_KEYS: List[str] = [
    "distance_to_industry_m",
    "distance_to_refinery_m",
    "distance_to_oil_gas_m",
    "distance_to_mining_m",
    "distance_to_agriculture_m",
    "distance_to_forest_m",
    "distance_to_power_plant_m",
]

# Canonical SpatialContext schema field names (matching schemas/spatial_context.py)
SPATIAL_CONTEXT_FIELDS: List[str] = [
    "nearest_industry_m",
    "nearest_refinery_m",
    "nearest_oil_gas_m",
    "nearest_mining_m",
    "nearest_agriculture_m",
    "nearest_forest_m",
    "nearest_power_plant_m",
]

# ---------------------------------------------------------------------------
# 3. Canonical 7-Class Hotspot Classification Taxonomy
# Final classification labels used across weak supervision, ML, and evaluation.
# ---------------------------------------------------------------------------
CLASS_LABELS: List[str] = [
    "industrial_fire",         # Active fire associated with industrial infrastructure
    "gas_flare",               # Oil/gas flare or persistent thermal source
    "mining_activity",         # Thermal signature from mining/quarry operations
    "agricultural_burn",       # Seasonal crop-residue or stubble burning
    "forest_natural_fire",     # Wildfire or natural forest vegetation fire
    "industrial_process_heat", # Persistent heat from industrial process (non-fire)
    "unclassified",            # Unknown / insufficient evidence / requires verification
]

# Display-friendly human-readable labels for UI, dashboards, and reporting
CLASS_LABEL_DISPLAY: Dict[str, str] = {
    "industrial_fire":         "Industrial Fire",
    "gas_flare":               "Gas Flare / Persistent Thermal Source",
    "mining_activity":         "Mining Activity",
    "agricultural_burn":       "Agricultural Burn",
    "forest_natural_fire":     "Forest / Natural Fire",
    "industrial_process_heat": "Industrial Process Heat",
    "unclassified":            "Unknown / Requires Verification",
}

# Hex color palette for map GIS visualization and UI cards
CLASS_COLORS: Dict[str, str] = {
    "industrial_fire":         "#FF4500",   # OrangeRed
    "gas_flare":               "#FF8C00",   # DarkOrange
    "mining_activity":         "#8B4513",   # SaddleBrown
    "agricultural_burn":       "#DAA520",   # GoldenRod
    "forest_natural_fire":     "#228B22",   # ForestGreen
    "industrial_process_heat": "#DC143C",   # Crimson
    "unclassified":            "#808080",   # Gray
}

# ---------------------------------------------------------------------------
# 4. NASA FIRMS Detection Metadata & Satellite Configurations
# ---------------------------------------------------------------------------
CONFIDENCE_LEVELS: List[str] = [
    "high",
    "nominal",
    "low",
]

FIRMS_DATASETS: Dict[str, str] = {
    "VIIRS_SNPP_NRT":   "Suomi NPP VIIRS (375m, 2x/day)",
    "VIIRS_NOAA20_NRT": "NOAA-20 VIIRS (375m, 2x/day)",
    "VIIRS_NOAA21_NRT": "NOAA-21 VIIRS (375m, 2x/day)",
    "MODIS_NRT":        "MODIS Terra/Aqua (1km, 4x/day)",
}

FIRMS_TYPE_CODES: Dict[int, str] = {
    -1: "unknown",
    0:  "presumed_vegetation_fire",
    2:  "active_volcano",
    3:  "static_land_source",
    4:  "offshore",
}

# ---------------------------------------------------------------------------
# 5. Spatial Analysis Defaults and Proximity Thresholds
# Note: These thresholds serve as baseline defaults and will be calibrated
# in labeling_functions.py / weak supervision rules.
# ---------------------------------------------------------------------------
DEFAULT_RADIUS_METERS: int = 15_000       # Default OSM spatial search radius (15 km)
REFINERY_PROXIMITY_M: int = 2_000        # Baseline proximity to refinery (2 km)
OIL_GAS_PROXIMITY_M: int = 2_000         # Baseline proximity to oil/gas facility (2 km)
INDUSTRIAL_PROXIMITY_M: int = 2_000      # Baseline proximity to industrial zone (2 km) -- calibrated for FIRMS VIIRS detection distribution
POWER_PLANT_PROXIMITY_M: int = 5_000     # Baseline proximity to power plant (5 km)
MINING_PROXIMITY_M: int = 2_000          # Baseline proximity to mining site (2 km)

# ---------------------------------------------------------------------------
# Module Exports
# ---------------------------------------------------------------------------
__all__ = [
    "SITE_TYPES",
    "OSM_CATEGORIES",
    "DISTANCE_KEYS",
    "SPATIAL_CONTEXT_FIELDS",
    "CLASS_LABELS",
    "CLASS_LABEL_DISPLAY",
    "CLASS_COLORS",
    "CONFIDENCE_LEVELS",
    "FIRMS_DATASETS",
    "FIRMS_TYPE_CODES",
    "DEFAULT_RADIUS_METERS",
    "REFINERY_PROXIMITY_M",
    "OIL_GAS_PROXIMITY_M",
    "INDUSTRIAL_PROXIMITY_M",
    "POWER_PLANT_PROXIMITY_M",
    "MINING_PROXIMITY_M",
]
