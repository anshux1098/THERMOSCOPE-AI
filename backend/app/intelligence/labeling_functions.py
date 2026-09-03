"""
labeling_functions.py
Weak Supervision Labeling Functions (LFs) for THERMOSCOPE-AI (SIH26162).

Architecture:
- 13 independent, domain-expert Labeling Functions.
- Zero eager "unclassified" voting: LFs vote for a target class or ABSTAIN (None).
- Safe feature extraction supporting dicts, Pandas Series/DataFrames, and Pydantic models.
- Graceful missing-feature handling (returns ABSTAIN if features are missing/NaN/sentinel).
- Registry and voting aggregator utilities for weak supervision training and rule consensus.

Taxonomy (7 Canonical Classes):
1. industrial_fire
2. gas_flare
3. mining_activity
4. agricultural_burn
5. forest_natural_fire
6. industrial_process_heat
7. unclassified (Default fallback when all LFs abstain or tie)
"""
import sys
import math
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union
from collections import Counter

# Set stdout to UTF-8 if supported to prevent Windows cp1252 UnicodeEncodeError
if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

# Ensure app package is discoverable when executed directly
backend_dir = str(Path(__file__).resolve().parents[2])
root_dir = str(Path(__file__).resolve().parents[3])
for p in (backend_dir, root_dir):
    if p not in sys.path:
        sys.path.insert(0, p)

from app.core.constants import (
    CLASS_LABELS,
    INDUSTRIAL_PROXIMITY_M,
    REFINERY_PROXIMITY_M,
    OIL_GAS_PROXIMITY_M,
    MINING_PROXIMITY_M,
    POWER_PLANT_PROXIMITY_M,
)

# ---------------------------------------------------------------------------
# Constants & Label Definitions
# ---------------------------------------------------------------------------
ABSTAIN: Optional[str] = None

INDUSTRIAL_FIRE: str = "industrial_fire"
GAS_FLARE: str = "gas_flare"
MINING_ACTIVITY: str = "mining_activity"
AGRICULTURAL_BURN: str = "agricultural_burn"
FOREST_NATURAL_FIRE: str = "forest_natural_fire"
INDUSTRIAL_PROCESS_HEAT: str = "industrial_process_heat"
UNCLASSIFIED: str = "unclassified"


# ---------------------------------------------------------------------------
# Safe Feature Extraction Helpers
# ---------------------------------------------------------------------------
def is_missing(val: Any) -> bool:
    """
    Check if a feature value is missing, NaN, sentinel (999/inf), or empty.
    """
    if val is None:
        return True
    if isinstance(val, (int, float)):
        if math.isnan(val) or math.isinf(val):
            return True
        if val in (999, 999.0, -999, -999.0):
            return True
        return False
    if isinstance(val, str):
        return val.strip() == "" or val.strip().lower() in ("none", "nan", "null")
    return False


def get_feature(record: Any, *keys: str, default: Any = None) -> Any:
    """
    Safely extract a feature from various record types:
    - Dict / nested dict
    - Pandas Series / DataFrame row
    - Pydantic BaseModel (e.g. Hotspot, SpatialContext, HotspotAnalysis)
    - Python object with attributes
    """
    if record is None:
        return default

    # 1. Pydantic / Object attribute inspection
    for key in keys:
        if hasattr(record, key):
            val = getattr(record, key)
            if not is_missing(val):
                return val

    # 2. Check nested object structures (e.g. HotspotAnalysis with hotspot/spatial_context)
    for sub_obj_name in ("spatial_context", "hotspot", "categories", "summary_distances"):
        if hasattr(record, sub_obj_name):
            sub_obj = getattr(record, sub_obj_name)
            for key in keys:
                if isinstance(sub_obj, dict) and key in sub_obj:
                    val = sub_obj[key]
                    if not is_missing(val):
                        return val
                elif hasattr(sub_obj, key):
                    val = getattr(sub_obj, key)
                    if not is_missing(val):
                        return val

    # 3. Dict / Pandas Series key lookup
    if isinstance(record, dict) or hasattr(record, "get"):
        for key in keys:
            val = record.get(key)
            if not is_missing(val):
                return val

        # Check nested dictionary fields if present
        for sub_dict_name in ("spatial_context", "hotspot", "categories", "summary_distances"):
            sub_dict = record.get(sub_dict_name)
            if isinstance(sub_dict, dict):
                for key in keys:
                    val = sub_dict.get(key)
                    if not is_missing(val):
                        return val

    return default


def get_distance_meters(record: Any, category: str) -> Optional[float]:
    """
    Safely extract geodesic distance in METERS for a target category.
    Handles multiple naming conventions and converts kilometer values if needed.
    
    Supported categories:
    - 'industry'
    - 'refinery'
    - 'oil_gas'
    - 'mining'
    - 'agriculture'
    - 'forest'
    - 'power_plant'
    """
    # 1. Check canonical meters keys first
    meters_keys = [
        f"distance_to_{category}_m",
        f"nearest_{category}_m",
        f"dist_{category}_m",
    ]
    val = get_feature(record, *meters_keys)
    if val is not None:
        try:
            return float(val)
        except (ValueError, TypeError):
            pass

    # 2. Check category-specific aliases (meters or kilometers in existing datasets)
    alias_map: Dict[str, List[str]] = {
        "industry": [
            "dist_factory",
            "dist_industrial_zone",
            "dist_to_nearest_industrial_km",
            "distance_to_industry",
        ],
        "refinery": ["dist_refinery", "distance_to_refinery"],
        "oil_gas": ["dist_oil_gas", "distance_to_oil_gas"],
        "mining": ["dist_mining", "distance_to_mining"],
        "agriculture": ["dist_agriculture", "dist_cropland", "distance_to_agriculture"],
        "forest": ["dist_forest", "dist_woodland", "distance_to_forest"],
        "power_plant": ["dist_powerplant", "dist_power_plant", "distance_to_power_plant"],
    }

    aliases = alias_map.get(category, [])
    alias_val = get_feature(record, *aliases)
    if alias_val is not None:
        try:
            num = float(alias_val)
            # In processed datasets (e.g. classified_hotspots.csv), distances under 500
            # are in kilometers (with 999 as sentinel for missing).
            # If <= 500, it represents kilometers -> convert to meters.
            if 0.0 <= num <= 500.0:
                return round(num * 1000.0, 2)
            elif num > 500.0 and num < 999.0:
                return round(num, 2)
        except (ValueError, TypeError):
            pass

    return None


def get_frp(record: Any) -> Optional[float]:
    """Extract Fire Radiative Power (MW)."""
    val = get_feature(record, "frp", "frp_mean", "frp_max", "frp_median")
    if val is not None:
        try:
            return float(val)
        except (ValueError, TypeError):
            pass
    return None


def get_brightness(record: Any) -> Optional[float]:
    """Extract brightness temperature in Kelvin."""
    val = get_feature(record, "brightness", "bright_ti4", "bright_ti4_mean", "bright_ti5", "bright_t31")
    if val is not None:
        try:
            return float(val)
        except (ValueError, TypeError):
            pass
    return None


def get_confidence(record: Any) -> str:
    """Extract and normalize detection confidence ('high', 'nominal', 'low')."""
    val = get_feature(record, "confidence", "confidence_val", "high_conf_fraction")
    if val is None:
        return "nominal"
    s = str(val).strip().lower()
    if s in ("h", "high", "1", "1.0") or (isinstance(val, (int, float)) and val > 0.7):
        return "high"
    if s in ("l", "low", "0", "0.0") or (isinstance(val, (int, float)) and val < 0.3):
        return "low"
    return "nominal"


def is_night(record: Any) -> Optional[bool]:
    """Check if the detection occurred at night."""
    val = get_feature(record, "is_night", "daynight", "night_fraction")
    if val is None:
        return None
    if isinstance(val, bool):
        return val
    if isinstance(val, (int, float)):
        return val >= 0.5
    if isinstance(val, str):
        return val.strip().upper() == "N" or val.strip().lower() in ("true", "1", "night")
    return None


def get_firms_type(record: Any) -> Optional[int]:
    """
    Extract FIRMS type code:
    0 = presumed vegetation fire
    2 = active volcano
    3 = other static land source
    4 = offshore
    -1 = unknown / VIIRS
    """
    val = get_feature(record, "firms_type", "firms_type_mode")
    if val is not None:
        try:
            return int(float(val))
        except (ValueError, TypeError):
            pass
    return None


# ---------------------------------------------------------------------------
# 1. Industrial Fire Labeling Functions (3 Functions)
# ---------------------------------------------------------------------------
def lf_industry_high_frp(record: Any) -> Optional[str]:
    """
    LF 1: Industrial Fire - Proximity + High FRP
    IF hotspot is within 1,000m of an industrial facility
    AND FRP is high (>= 35.0 MW)
    -> VOTE: INDUSTRIAL_FIRE
    """
    dist_ind = get_distance_meters(record, "industry")
    frp = get_frp(record)

    if dist_ind is not None and frp is not None:
        if dist_ind <= INDUSTRIAL_PROXIMITY_M and frp >= 35.0:
            return INDUSTRIAL_FIRE

    return ABSTAIN


def lf_factory_proximity_thermal(record: Any) -> Optional[str]:
    """
    LF 2: Industrial Fire - Factory Proximity + Elevated Brightness
    IF hotspot is within 1,000m of a factory
    AND brightness temperature is elevated (>= 330.0 K)
    AND confidence is nominal or high
    -> VOTE: INDUSTRIAL_FIRE
    """
    dist_ind = get_distance_meters(record, "industry")
    brightness = get_brightness(record)
    conf = get_confidence(record)

    if dist_ind is not None and brightness is not None:
        if dist_ind <= INDUSTRIAL_PROXIMITY_M and brightness >= 330.0 and conf != "low":
            return INDUSTRIAL_FIRE

    return ABSTAIN


def lf_industrial_zone_cluster(record: Any) -> Optional[str]:
    """
    LF 3: Industrial Fire - Industrial Zone Spatial Density
    IF hotspot is within 1,500m of an industrial zone
    AND has multiple industrial sites within 5km (or has_industrial_2km indicator)
    AND FRP is significant (>= 20.0 MW)
    -> VOTE: INDUSTRIAL_FIRE
    """
    dist_ind = get_distance_meters(record, "industry")
    frp = get_frp(record)
    has_ind_2km = get_feature(record, "has_industrial_2km", "has_industrial_within_2km")
    count_ind_5km = get_feature(record, "count_ind_5km", "num_industrial_sites_within_5km", default=0)

    is_dense_industrial = (has_ind_2km == 1) or (count_ind_5km is not None and count_ind_5km >= 2)

    if dist_ind is not None and frp is not None:
        if dist_ind <= 1500.0 and is_dense_industrial and frp >= 20.0:
            return INDUSTRIAL_FIRE

    return ABSTAIN


# ---------------------------------------------------------------------------
# 2. Gas Flare Labeling Functions (2 Functions)
# ---------------------------------------------------------------------------
def lf_oil_gas_proximity(record: Any) -> Optional[str]:
    """
    LF 4: Gas Flare - Oil & Gas Infrastructure Proximity
    IF hotspot is within 1,500m of an oil/gas extraction or processing facility
    -> VOTE: GAS_FLARE
    """
    dist_oil_gas = get_distance_meters(record, "oil_gas")

    if dist_oil_gas is not None and dist_oil_gas <= OIL_GAS_PROXIMITY_M:
        return GAS_FLARE

    return ABSTAIN


def lf_refinery_persistent_heat(record: Any) -> Optional[str]:
    """
    LF 5: Gas Flare - Refinery + Nighttime / Static Signature
    Refineries often flare continuously across day and night.
    IF hotspot is within 2,000m of a refinery
    AND (detected at night OR persistence ratio >= 0.3 OR static land source firms_type == 3)
    -> VOTE: GAS_FLARE
    """
    dist_refinery = get_distance_meters(record, "refinery")
    night = is_night(record)
    firms_type = get_firms_type(record)
    persistence = get_feature(record, "persistence_ratio", default=0.0)

    is_persistent = (night is True) or (firms_type == 3) or (persistence is not None and persistence >= 0.3)

    if dist_refinery is not None and dist_refinery <= REFINERY_PROXIMITY_M and is_persistent:
        return GAS_FLARE

    return ABSTAIN


# ---------------------------------------------------------------------------
# 3. Mining Activity Labeling Functions (2 Functions)
# ---------------------------------------------------------------------------
def lf_mining_proximity(record: Any) -> Optional[str]:
    """
    LF 6: Mining Activity - Direct Mining Site / Quarry Proximity
    IF hotspot is within 2,000m of a quarry, coal mine, or mineral extraction site
    -> VOTE: MINING_ACTIVITY
    """
    dist_mining = get_distance_meters(record, "mining")

    if dist_mining is not None and dist_mining <= MINING_PROXIMITY_M:
        return MINING_ACTIVITY

    return ABSTAIN


def lf_mining_high_confidence(record: Any) -> Optional[str]:
    """
    LF 7: Mining Activity - Mining Region + High Detection Confidence
    IF hotspot is within 3,000m of a mining site
    AND confidence is high
    AND FRP is moderate (>= 10.0 MW)
    -> VOTE: MINING_ACTIVITY
    """
    dist_mining = get_distance_meters(record, "mining")
    conf = get_confidence(record)
    frp = get_frp(record)

    if dist_mining is not None and dist_mining <= 3000.0 and conf == "high":
        if frp is not None and frp >= 10.0:
            return MINING_ACTIVITY

    return ABSTAIN


# ---------------------------------------------------------------------------
# 4. Agricultural Burn Labeling Functions (2 Functions)
# ---------------------------------------------------------------------------
def lf_agriculture_proximity(record: Any) -> Optional[str]:
    """
    LF 8: Agricultural Burn - Cropland Proximity + Moderate Thermal Signal
    IF agricultural/farmland is within 1,000m
    AND FRP is moderate/low (< 50.0 MW) typical of stubble / crop burning
    AND not right inside heavy industry (> 1,500m from industrial sites)
    -> VOTE: AGRICULTURAL_BURN
    """
    dist_agri = get_distance_meters(record, "agriculture")
    dist_ind = get_distance_meters(record, "industry")
    frp = get_frp(record)

    # Safe missing feature handling: if dist_agriculture is missing, safely ABSTAIN
    if dist_agri is None:
        return ABSTAIN

    if dist_agri <= 1000.0:
        if (dist_ind is None or dist_ind > 1500.0) and (frp is None or frp < 50.0):
            return AGRICULTURAL_BURN

    return ABSTAIN


def lf_vegetation_fire_low_frp(record: Any) -> Optional[str]:
    """
    LF 9: Agricultural Burn - Presumed Vegetation Fire + Low Intensity in Open Terrain
    IF FIRMS classifies type as presumed vegetation fire (firms_type == 0)
    AND FRP is low (< 25.0 MW)
    AND hotspot is far from industrial facilities (> 2,500m)
    -> VOTE: AGRICULTURAL_BURN
    """
    firms_type = get_firms_type(record)
    frp = get_frp(record)
    dist_ind = get_distance_meters(record, "industry")

    if firms_type == 0 and frp is not None and frp < 25.0:
        if dist_ind is None or dist_ind >= 2500.0:
            return AGRICULTURAL_BURN

    return ABSTAIN


# ---------------------------------------------------------------------------
# 5. Forest / Natural Fire Labeling Functions (2 Functions)
# ---------------------------------------------------------------------------
def lf_forest_proximity(record: Any) -> Optional[str]:
    """
    LF 10: Forest Fire - Forest / Woodland Proximity + Isolated from Industry
    IF hotspot is within 1,500m of forest / woodland
    AND far from industrial zones (> 3,000m)
    -> VOTE: FOREST_NATURAL_FIRE
    """
    dist_forest = get_distance_meters(record, "forest")
    dist_ind = get_distance_meters(record, "industry")

    # Safe missing feature handling: if dist_forest is missing, safely ABSTAIN
    if dist_forest is None:
        return ABSTAIN

    if dist_forest <= 1500.0:
        if dist_ind is None or dist_ind >= 3000.0:
            return FOREST_NATURAL_FIRE

    return ABSTAIN


def lf_vegetation_high_frp_isolated(record: Any) -> Optional[str]:
    """
    LF 11: Forest Fire - High FRP Vegetation Fire Isolated in Wilderness
    Wildfires produce large thermal signatures across dense fuel.
    IF FRP is high (>= 30.0 MW)
    AND (firms_type == 0 OR distant from all industry > 4,000m)
    AND not in an agricultural zone
    -> VOTE: FOREST_NATURAL_FIRE
    """
    frp = get_frp(record)
    firms_type = get_firms_type(record)
    dist_ind = get_distance_meters(record, "industry")
    dist_agri = get_distance_meters(record, "agriculture")

    if frp is not None and frp >= 30.0:
        is_isolated = (dist_ind is None or dist_ind >= 4000.0)
        not_agri = (dist_agri is None or dist_agri > 1000.0)

        if is_isolated and not_agri and (firms_type == 0 or firms_type is None):
            return FOREST_NATURAL_FIRE

    return ABSTAIN


# ---------------------------------------------------------------------------
# 6. Industrial Process Heat Labeling Functions (2 Functions)
# ---------------------------------------------------------------------------
def lf_static_industrial_heat(record: Any) -> Optional[str]:
    """
    LF 12: Process Heat - FIRMS Static Land Source + Industrial Proximity
    FIRMS type 3 indicates a known static land source (e.g. kiln, blast furnace, power boiler).
    IF firms_type == 3 (static land source)
    AND hotspot is within 2,500m of an industrial site
    -> VOTE: INDUSTRIAL_PROCESS_HEAT
    """
    firms_type = get_firms_type(record)
    dist_ind = get_distance_meters(record, "industry")

    if firms_type == 3:
        if dist_ind is not None and dist_ind <= 2500.0:
            return INDUSTRIAL_PROCESS_HEAT

    return ABSTAIN


def lf_nighttime_low_frp_industrial(record: Any) -> Optional[str]:
    """
    LF 13: Process Heat - Nighttime Steady Low FRP Industrial Signature
    Industrial processes (cement kilns, metal smelters) emit steady low heat at night
    without the violent flare-ups of active industrial fires.
    IF detected at night
    AND FRP is low/steady (<= 15.0 MW)
    AND brightness is moderate (>= 310.0 K)
    AND hotspot is within 1,200m of an industrial site
    -> VOTE: INDUSTRIAL_PROCESS_HEAT
    """
    night = is_night(record)
    frp = get_frp(record)
    brightness = get_brightness(record)
    dist_ind = get_distance_meters(record, "industry")

    if night is True and frp is not None and frp <= 15.0:
        if brightness is not None and brightness >= 310.0:
            if dist_ind is not None and dist_ind <= 1200.0:
                return INDUSTRIAL_PROCESS_HEAT

    return ABSTAIN


# ---------------------------------------------------------------------------
# Labeling Function Registry
# ---------------------------------------------------------------------------
ALL_LABELING_FUNCTIONS: List[Callable[[Any], Optional[str]]] = [
    # 🏭 Industrial Fire (3 LFs)
    lf_industry_high_frp,
    lf_factory_proximity_thermal,
    lf_industrial_zone_cluster,

    # 🔥 Gas Flare (2 LFs)
    lf_oil_gas_proximity,
    lf_refinery_persistent_heat,

    # ⛏️ Mining Activity (2 LFs)
    lf_mining_proximity,
    lf_mining_high_confidence,

    # 🌾 Agricultural Burn (2 LFs)
    lf_agriculture_proximity,
    lf_vegetation_fire_low_frp,

    # 🌲 Forest / Natural Fire (2 LFs)
    lf_forest_proximity,
    lf_vegetation_high_frp_isolated,

    # ♨️ Industrial Process Heat (2 LFs)
    lf_static_industrial_heat,
    lf_nighttime_low_frp_industrial,
]

LABELING_FUNCTION_MAP: Dict[str, Callable[[Any], Optional[str]]] = {
    lf.__name__: lf for lf in ALL_LABELING_FUNCTIONS
}


# ---------------------------------------------------------------------------
# Execution & Aggregation Utilities
# ---------------------------------------------------------------------------
def apply_labeling_functions(record: Any) -> Dict[str, Optional[str]]:
    """
    Apply all registered labeling functions to a single hotspot record.
    Returns a dictionary mapping function_name -> voted_class_or_None.
    """
    return {
        lf.__name__: lf(record)
        for lf in ALL_LABELING_FUNCTIONS
    }


def aggregate_votes(votes: Union[Dict[str, Optional[str]], List[Optional[str]]]) -> str:
    """
    Aggregate LF votes into a single consensus prediction using majority voting.
    
    Rules:
    1. Filters out ABSTAIN (None) votes.
    2. If all LFs abstain -> returns 'unclassified'.
    3. Returns the most frequent non-abstain class.
    4. On a tie, returns the highest-priority non-abstain class or 'unclassified'.
    """
    if isinstance(votes, dict):
        active_votes = [v for v in votes.values() if v is not None]
    elif isinstance(votes, (list, tuple)):
        active_votes = [v for v in votes if v is not None]
    else:
        active_votes = []

    if not active_votes:
        return UNCLASSIFIED

    counts = Counter(active_votes)
    top_class, _ = counts.most_common(1)[0]
    return top_class


# ---------------------------------------------------------------------------
# Module Verification & Demonstration Test Harness
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("=" * 72)
    print("THERMOSCOPE-AI: LABELING FUNCTIONS VERIFICATION HARNESS")
    print("=" * 72)
    print(f"[*] Total Registered Labeling Functions: {len(ALL_LABELING_FUNCTIONS)}")
    for i, lf in enumerate(ALL_LABELING_FUNCTIONS, 1):
        print(f"  {i:02d}. {lf.__name__}")

    print("\n" + "=" * 72)
    print("RUNNING SYNTHETIC TEST CASES FOR ALL 7 TAXONOMY CLASSES")
    print("=" * 72)

    test_cases = [
        {
            "name": "Case 1: Steel Rolling Mill Fire (High FRP + Factory)",
            "hotspot": {
                "latitude": 30.3165,
                "longitude": 78.0322,
                "frp": 65.0,
                "brightness": 345.0,
                "confidence": "high",
                "distance_to_industry_m": 350.0,
                "distance_to_forest_m": 4200.0,
                "has_industrial_2km": 1,
            },
            "expected": INDUSTRIAL_FIRE,
        },
        {
            "name": "Case 2: Offshore / Onshore Gas Flare",
            "hotspot": {
                "latitude": 21.6500,
                "longitude": 72.5000,
                "frp": 28.0,
                "brightness": 332.0,
                "is_night": True,
                "distance_to_oil_gas_m": 450.0,
                "distance_to_refinery_m": 1200.0,
            },
            "expected": GAS_FLARE,
        },
        {
            "name": "Case 3: Open-Cast Coal Mine Thermal Signature",
            "hotspot": {
                "latitude": 23.8000,
                "longitude": 86.4000,
                "frp": 18.5,
                "brightness": 322.0,
                "confidence": "high",
                "distance_to_mining_m": 600.0,
                "distance_to_industry_m": 8500.0,
            },
            "expected": MINING_ACTIVITY,
        },
        {
            "name": "Case 4: Stubble / Agricultural Residue Burn",
            "hotspot": {
                "latitude": 30.8000,
                "longitude": 75.8000,
                "frp": 14.0,
                "brightness": 318.0,
                "firms_type": 0,
                "distance_to_agriculture_m": 250.0,
                "distance_to_industry_m": 9200.0,
            },
            "expected": AGRICULTURAL_BURN,
        },
        {
            "name": "Case 5: Wildfire / Deep Forest Fire",
            "hotspot": {
                "latitude": 30.5000,
                "longitude": 79.1000,
                "frp": 85.0,
                "brightness": 355.0,
                "firms_type": 0,
                "distance_to_forest_m": 400.0,
                "distance_to_industry_m": 18000.0,
            },
            "expected": FOREST_NATURAL_FIRE,
        },
        {
            "name": "Case 6: Industrial Blast Furnace / Kiln (Process Heat)",
            "hotspot": {
                "latitude": 22.8000,
                "longitude": 86.2000,
                "frp": 11.0,
                "brightness": 315.0,
                "firms_type": 3,
                "is_night": True,
                "distance_to_industry_m": 450.0,
            },
            "expected": INDUSTRIAL_PROCESS_HEAT,
        },
        {
            "name": "Case 7: Sparse / Isolated Hotspot (All Abstain -> Unclassified)",
            "hotspot": {
                "latitude": 15.0000,
                "longitude": 75.0000,
                "frp": 4.5,
                "brightness": 302.0,
                "confidence": "low",
                "distance_to_industry_m": 45000.0,
            },
            "expected": UNCLASSIFIED,
        },
    ]

    all_passed = True
    for case in test_cases:
        votes = apply_labeling_functions(case["hotspot"])
        consensus = aggregate_votes(votes)
        active = [f"{k} -> {v}" for k, v in votes.items() if v is not None]
        passed = consensus == case["expected"]
        if not passed:
            all_passed = False

        status_mark = "[PASS]" if passed else "[FAIL]"
        print(f"\n{status_mark} | {case['name']}")
        print(f"       Expected  : {case['expected']}")
        print(f"       Consensus : {consensus}")
        print(f"       Active LFs: {len(active)} voted ({', '.join(active) if active else 'All Abstained'})")

    print("\n" + "=" * 72)
    if all_passed:
        print("[*] ALL TEST CASES PASSED SUCCESSFULLY!")
    else:
        print("[!] SOME TEST CASES FAILED!")
    print("=" * 72)
