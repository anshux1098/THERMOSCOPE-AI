"""
labeling_functions.py
Weak Supervision Labeling Functions (LFs) for THERMOSCOPE-AI (SIH26162).

Architecture:
- 13 independent, explainable, domain-expert Labeling Functions (LFs).
- Each LF behaves as an independent detector voting for a specific canonical target class or ABSTAIN (None).
- Zero eager 'unclassified' voting (unclassified fallback is handled at the aggregation layer).
- Conservative, evidence-based rules: When evidence is insufficient or missing -> ABSTAIN.
- Strict Distance Unit Discipline: Expects distances in METERS via canonical schema fields.
  (No arbitrary heuristic unit-guessing).
- Defensive feature extraction supporting Python dicts, Pandas Series/DataFrames, and Pydantic models.

Taxonomy (7 Canonical Classes matching app.core.constants.CLASS_LABELS):
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
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Union

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
# Label Constants & ABSTAIN Definition
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
# Heuristic Threshold Constants for Weak Supervision
# (Note: These serve as baseline domain heuristics and will be calibrated
# against the enriched spatial dataset during model training.)
# ---------------------------------------------------------------------------
# Spatial Proximity Thresholds (in METERS)
THRESHOLD_INDUSTRY_PROXIMITY_M: float = float(INDUSTRIAL_PROXIMITY_M)   # 1,000 m
THRESHOLD_REFINERY_PROXIMITY_M: float = float(REFINERY_PROXIMITY_M)   # 2,000 m
THRESHOLD_OIL_GAS_PROXIMITY_M: float = float(OIL_GAS_PROXIMITY_M)     # 2,000 m
THRESHOLD_MINING_PROXIMITY_M: float = float(MINING_PROXIMITY_M)       # 2,000 m
THRESHOLD_AGRICULTURE_PROXIMITY_M: float = 15_000.0                   # 15,000 m (15 km satellite buffer)
THRESHOLD_FOREST_PROXIMITY_M: float = 15_000.0                        # 15,000 m (15 km satellite buffer)
THRESHOLD_ISOLATED_FROM_INDUSTRY_M: float = 1_500.0                   # 1,500 m
THRESHOLD_DEEP_ISOLATION_INDUSTRY_M: float = 3_000.0                  # 3,000 m

# Thermal Intensity Thresholds
FRP_VERY_HIGH_MW: float = 35.0         # High-intensity industrial or wildfire event
FRP_MODERATE_MW: float = 15.0          # Moderate thermal activity
FRP_LOW_STEADY_MW: float = 15.0        # Low steady process heat / stubble burn
FRP_AGRI_MAX_MW: float = 45.0          # Upper bound for agricultural stubble burns
BRIGHTNESS_ELEVATED_K: float = 330.0   # Elevated Kelvin brightness temperature
BRIGHTNESS_MODERATE_K: float = 310.0   # Moderate steady thermal signature


# ---------------------------------------------------------------------------
# Safe Feature Extraction Helpers
# ---------------------------------------------------------------------------
def is_missing(val: Any) -> bool:
    """
    Check if a feature value is missing, None, NaN, infinite, sentinel (999), or empty string.
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
    if isinstance(record, (dict, Mapping)) or hasattr(record, "get"):
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


def get_numeric_feature(record: Any, *keys: str, default: Optional[float] = None) -> Optional[float]:
    """
    Safely extract a numeric feature and convert to float.
    Returns None if missing, non-numeric, or sentinel.
    """
    val = get_feature(record, *keys)
    if val is None or is_missing(val):
        return default
    try:
        return float(val)
    except (ValueError, TypeError):
        return default


def get_distance_meters(record: Any, category: str) -> Optional[float]:
    """
    Safely extract geodesic distance in METERS for a target category.
    Strictly expects canonical meter fields or explicit meter aliases.
    Does not guess units based on magnitude.

    Supported categories:
    - 'industry'
    - 'refinery'
    - 'oil_gas'
    - 'mining'
    - 'agriculture'
    - 'forest'
    - 'power_plant'
    """
    # 1. Check canonical meters keys
    meters_keys = [
        f"distance_to_{category}_m",
        f"nearest_{category}_m",
        f"dist_{category}_m",
    ]
    val = get_numeric_feature(record, *meters_keys)
    if val is not None:
        return val

    # 2. Check category-specific explicit aliases
    alias_map: Dict[str, List[str]] = {
        "industry": ["dist_factory_m", "dist_industrial_zone_m", "distance_to_industry"],
        "refinery": ["distance_to_refinery"],
        "oil_gas": ["distance_to_oil_gas"],
        "mining": ["distance_to_mining"],
        "agriculture": ["distance_to_agriculture", "dist_agriculture_m", "dist_cropland_m"],
        "forest": ["distance_to_forest", "dist_forest_m", "dist_woodland_m"],
        "power_plant": ["distance_to_power_plant", "dist_powerplant_m", "dist_power_plant_m"],
    }
    aliases = alias_map.get(category, [])
    val = get_numeric_feature(record, *aliases)
    return val


def get_frp(record: Any) -> Optional[float]:
    """Extract Fire Radiative Power (MW)."""
    return get_numeric_feature(record, "frp", "frp_mean", "frp_max", "frp_median")


def get_brightness(record: Any) -> Optional[float]:
    """Extract brightness temperature in Kelvin."""
    return get_numeric_feature(record, "brightness", "bright_ti4", "bright_ti4_mean", "bright_ti5", "bright_t31")


def get_confidence(record: Any) -> str:
    """Extract and normalize detection confidence ('high', 'nominal', 'low')."""
    val = get_feature(record, "confidence", "confidence_val", "high_conf_fraction")
    if val is None or is_missing(val):
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
    if val is None or is_missing(val):
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
    val = get_numeric_feature(record, "firms_type", "firms_type_mode")
    if val is not None:
        try:
            return int(val)
        except (ValueError, TypeError):
            pass
    return None


# ---------------------------------------------------------------------------
# 1. 🏭 Industrial Fire Labeling Functions (3 Functions)
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
        if dist_ind <= THRESHOLD_INDUSTRY_PROXIMITY_M and frp >= FRP_VERY_HIGH_MW:
            return INDUSTRIAL_FIRE

    return ABSTAIN


def lf_factory_proximity_thermal(record: Any) -> Optional[str]:
    """
    LF 2: Industrial Fire - Factory Proximity + Elevated Brightness
    IF hotspot is within 1,000m of industrial infrastructure
    AND brightness temperature is elevated (>= 330.0 K)
    AND confidence is nominal or high
    -> VOTE: INDUSTRIAL_FIRE
    """
    dist_ind = get_distance_meters(record, "industry")
    brightness = get_brightness(record)
    conf = get_confidence(record)

    if dist_ind is not None and brightness is not None:
        if dist_ind <= THRESHOLD_INDUSTRY_PROXIMITY_M and brightness >= BRIGHTNESS_ELEVATED_K and conf != "low":
            return INDUSTRIAL_FIRE

    return ABSTAIN


def lf_industrial_zone_cluster(record: Any) -> Optional[str]:
    """
    LF 3: Industrial Fire - Industrial Zone Spatial Density
    IF hotspot is within 1,500m of an industrial site
    AND has multiple industrial sites within 5km (or has_industrial_2km indicator)
    AND FRP is significant (>= 20.0 MW)
    -> VOTE: INDUSTRIAL_FIRE
    """
    dist_ind = get_distance_meters(record, "industry")
    frp = get_frp(record)
    has_ind_2km = get_feature(record, "has_industrial_2km", "has_industrial_within_2km")
    count_ind_5km = get_numeric_feature(record, "count_ind_5km", "num_industrial_sites_within_5km", default=0.0)

    is_dense_industrial = (has_ind_2km == 1) or (count_ind_5km is not None and count_ind_5km >= 2)

    if dist_ind is not None and frp is not None:
        if dist_ind <= 1500.0 and is_dense_industrial and frp >= 20.0:
            return INDUSTRIAL_FIRE

    return ABSTAIN


# ---------------------------------------------------------------------------
# 2. 🛢️ Gas Flare Labeling Functions (2 Functions)
# ---------------------------------------------------------------------------
def lf_oil_gas_flare(record: Any) -> Optional[str]:
    """
    LF 4: Gas Flare - Oil & Gas Infrastructure Proximity + Flare Evidence
    Proximity alone is not enough; requires nighttime detection, static type 3,
    or steady flare thermal signal (FRP >= 15.0 MW).
    IF hotspot is within 2,000m of an oil/gas facility
    AND (nighttime detection OR static land source firms_type == 3 OR FRP >= 15.0 MW)
    -> VOTE: GAS_FLARE
    """
    dist_oil_gas = get_distance_meters(record, "oil_gas")
    if dist_oil_gas is None or dist_oil_gas > THRESHOLD_OIL_GAS_PROXIMITY_M:
        return ABSTAIN

    night = is_night(record)
    firms_type = get_firms_type(record)
    frp = get_frp(record)

    has_flare_evidence = (night is True) or (firms_type == 3) or (frp is not None and frp >= FRP_MODERATE_MW)
    if has_flare_evidence:
        return GAS_FLARE

    return ABSTAIN


def lf_refinery_flare(record: Any) -> Optional[str]:
    """
    LF 5: Gas Flare - Refinery Proximity + Nighttime / Static Flare Signature
    Refineries often flare continuously across day and night.
    IF hotspot is within 2,000m of a refinery
    AND (detected at night OR static land source firms_type == 3 OR persistence ratio >= 0.3)
    -> VOTE: GAS_FLARE
    """
    dist_refinery = get_distance_meters(record, "refinery")
    if dist_refinery is None or dist_refinery > THRESHOLD_REFINERY_PROXIMITY_M:
        return ABSTAIN

    night = is_night(record)
    firms_type = get_firms_type(record)
    persistence = get_numeric_feature(record, "persistence_ratio", default=0.0)

    is_persistent_or_flare = (night is True) or (firms_type == 3) or (persistence is not None and persistence >= 0.3)
    if is_persistent_or_flare:
        return GAS_FLARE

    return ABSTAIN


# ---------------------------------------------------------------------------
# 3. ⛏️ Mining Activity Labeling Functions (2 Functions)
# ---------------------------------------------------------------------------
def lf_mining_thermal_activity(record: Any) -> Optional[str]:
    """
    LF 6: Mining Activity - Direct Mining Site Proximity + Thermal Signal
    Mining proximity alone is not enough; requires supporting thermal evidence
    and non-low confidence.
    IF hotspot is within 2,000m of a quarry, coal mine, or mineral extraction site
    AND (FRP >= 10.0 MW OR Brightness >= 315.0 K)
    AND confidence is nominal or high
    -> VOTE: MINING_ACTIVITY
    """
    dist_mining = get_distance_meters(record, "mining")
    if dist_mining is None or dist_mining > THRESHOLD_MINING_PROXIMITY_M:
        return ABSTAIN

    frp = get_frp(record)
    brightness = get_brightness(record)
    conf = get_confidence(record)

    has_thermal_evidence = (frp is not None and frp >= 10.0) or (brightness is not None and brightness >= BRIGHTNESS_MODERATE_K)
    if has_thermal_evidence and conf != "low":
        return MINING_ACTIVITY

    return ABSTAIN


def lf_mining_high_confidence(record: Any) -> Optional[str]:
    """
    LF 7: Mining Activity - Mining Region + High Confidence + Away from Heavy Factories
    IF hotspot is within 3,000m of a mining site
    AND confidence is high
    AND FRP is significant (>= 15.0 MW)
    AND not directly inside a dense factory/industrial site (> 1,500m from heavy industry)
    -> VOTE: MINING_ACTIVITY
    """
    dist_mining = get_distance_meters(record, "mining")
    if dist_mining is None or dist_mining > 3000.0:
        return ABSTAIN

    conf = get_confidence(record)
    frp = get_frp(record)
    dist_ind = get_distance_meters(record, "industry")

    if conf == "high" and frp is not None and frp >= FRP_MODERATE_MW:
        if dist_ind is None or dist_ind >= THRESHOLD_ISOLATED_FROM_INDUSTRY_M:
            return MINING_ACTIVITY

    return ABSTAIN


# ---------------------------------------------------------------------------
# 4. 🌾 Agricultural Burn Labeling Functions (2 Functions)
# ---------------------------------------------------------------------------
def lf_agriculture_vegetation_fire(record: Any) -> Optional[str]:
    """
    LF 8: Agricultural Burn - Cropland Proximity + Vegetation Fire Signal
    MUST require actual agriculture evidence (distance_to_agriculture_m).
    If agriculture context is missing -> ABSTAIN.
    """
    dist_agri = get_distance_meters(record, "agriculture")
    if dist_agri is None or dist_agri >= 45000.0:  # Strictly ABSTAIN if missing or sentinel
        return ABSTAIN

    if dist_agri <= THRESHOLD_AGRICULTURE_PROXIMITY_M:
        frp = get_frp(record)
        firms_type = get_firms_type(record)
        dist_ind = get_distance_meters(record, "industry")

        is_vegetation = (firms_type == 0) or (dist_ind is None or dist_ind >= 45000.0 or dist_ind >= THRESHOLD_ISOLATED_FROM_INDUSTRY_M or dist_agri < dist_ind)
        if is_vegetation and (frp is None or frp <= FRP_AGRI_MAX_MW):
            return AGRICULTURAL_BURN

    return ABSTAIN


def lf_agriculture_burn_context(record: Any) -> Optional[str]:
    """
    LF 9: Agricultural Burn - Farmland Proximity + Moderate Stubble Thermal Band
    MUST require actual agriculture evidence (distance_to_agriculture_m).
    If agriculture context is missing -> ABSTAIN.
    """
    dist_agri = get_distance_meters(record, "agriculture")
    if dist_agri is None or dist_agri >= 45000.0:  # Strictly ABSTAIN if missing or sentinel
        return ABSTAIN

    if dist_agri <= THRESHOLD_AGRICULTURE_PROXIMITY_M:
        frp = get_frp(record)
        brightness = get_brightness(record)
        dist_ind = get_distance_meters(record, "industry")

        is_stubble_thermal = (
            (frp is not None and 5.0 <= frp <= FRP_VERY_HIGH_MW)
            and (brightness is None or (BRIGHTNESS_MODERATE_K <= brightness <= 345.0))
        )
        is_isolated_ind = (dist_ind is None or dist_ind >= 45000.0 or dist_ind >= 1500.0 or dist_agri < dist_ind)

        if is_stubble_thermal and is_isolated_ind:
            return AGRICULTURAL_BURN

    return ABSTAIN


# ---------------------------------------------------------------------------
# 5. 🌲 Forest / Natural Fire Labeling Functions (2 Functions)
# ---------------------------------------------------------------------------
def lf_forest_vegetation_fire(record: Any) -> Optional[str]:
    """
    LF 10: Forest Fire - Forest / Woodland Proximity + Isolated Vegetation Fire
    MUST require actual forest evidence (distance_to_forest_m).
    If forest context is missing -> ABSTAIN.
    """
    dist_forest = get_distance_meters(record, "forest")
    if dist_forest is None or dist_forest >= 45000.0:  # Strictly ABSTAIN if missing or sentinel
        return ABSTAIN

    if dist_forest <= THRESHOLD_FOREST_PROXIMITY_M:
        firms_type = get_firms_type(record)
        frp = get_frp(record)
        dist_ind = get_distance_meters(record, "industry")

        is_veg_fire = (firms_type == 0) or (firms_type is None)
        is_isolated = (dist_ind is None or dist_ind >= 45000.0 or dist_ind >= THRESHOLD_ISOLATED_FROM_INDUSTRY_M or dist_forest < dist_ind)

        if is_veg_fire and is_isolated and (frp is None or frp >= 5.0):
            return FOREST_NATURAL_FIRE

    return ABSTAIN


def lf_strong_forest_fire(record: Any) -> Optional[str]:
    """
    LF 11: Forest Fire - Forest Proximity + High-Intensity Wildfire Signal
    MUST require actual forest evidence (distance_to_forest_m).
    If forest context is missing -> ABSTAIN.
    """
    dist_forest = get_distance_meters(record, "forest")
    if dist_forest is None or dist_forest >= 45000.0:  # Strictly ABSTAIN if missing or sentinel
        return ABSTAIN

    if dist_forest <= THRESHOLD_FOREST_PROXIMITY_M:
        frp = get_frp(record)
        dist_ind = get_distance_meters(record, "industry")
        dist_agri = get_distance_meters(record, "agriculture")

        if frp is not None and frp >= 15.0:
            is_isolated_ind = (dist_ind is None or dist_ind >= 45000.0 or dist_ind >= 2000.0 or dist_forest < dist_ind)
            is_not_agri = (dist_agri is None or dist_agri >= 45000.0 or dist_forest < dist_agri)

            if is_isolated_ind or is_not_agri:
                return FOREST_NATURAL_FIRE

    return ABSTAIN


# ---------------------------------------------------------------------------
# 6. ♨️ Industrial Process Heat Labeling Functions (2 Functions)
# ---------------------------------------------------------------------------
def lf_static_industrial_heat(record: Any) -> Optional[str]:
    """
    LF 12: Process Heat - Static Industrial Heat Source (Kiln / Boiler / Smelter)
    FIRMS type 3 indicates a known static land source.
    Distinguishes steady process heat from violent industrial fires.
    IF firms_type == 3 (static land source)
    AND hotspot is within 2,000m of an industrial site
    AND FRP is moderate/low (<= 25.0 MW)
    -> VOTE: INDUSTRIAL_PROCESS_HEAT
    """
    firms_type = get_firms_type(record)
    dist_ind = get_distance_meters(record, "industry")
    frp = get_frp(record)

    if firms_type == 3:
        if dist_ind is not None and dist_ind <= THRESHOLD_INDUSTRY_PROXIMITY_M * 2.0:
            if frp is None or frp <= 25.0:
                return INDUSTRIAL_PROCESS_HEAT

    return ABSTAIN


def lf_nighttime_process_heat(record: Any) -> Optional[str]:
    """
    LF 13: Process Heat - Nighttime Steady Low-FRP Industrial Signature
    Industrial continuous processes emit steady, low heat at night without
    the large flare-ups of active industrial fires.
    IF detected at night
    AND FRP is steady/low (<= 15.0 MW)
    AND brightness is moderate (>= 310.0 K)
    AND hotspot is within 1,200m of an industrial site
    -> VOTE: INDUSTRIAL_PROCESS_HEAT
    """
    night = is_night(record)
    frp = get_frp(record)
    brightness = get_brightness(record)
    dist_ind = get_distance_meters(record, "industry")

    if night is True and frp is not None and frp <= FRP_LOW_STEADY_MW:
        if brightness is not None and brightness >= BRIGHTNESS_MODERATE_K:
            if dist_ind is not None and dist_ind <= 1200.0:
                return INDUSTRIAL_PROCESS_HEAT

    return ABSTAIN


# ---------------------------------------------------------------------------
# 7. Labeling Function Registry
# ---------------------------------------------------------------------------
ALL_LABELING_FUNCTIONS: List[Callable[[Any], Optional[str]]] = [
    # 🏭 Industrial Fire (3 LFs)
    lf_industry_high_frp,
    lf_factory_proximity_thermal,
    lf_industrial_zone_cluster,

    # 🛢️ Gas Flare (2 LFs)
    lf_oil_gas_flare,
    lf_refinery_flare,

    # ⛏️ Mining Activity (2 LFs)
    lf_mining_thermal_activity,
    lf_mining_high_confidence,

    # 🌾 Agricultural Burn (2 LFs)
    lf_agriculture_vegetation_fire,
    lf_agriculture_burn_context,

    # 🌲 Forest / Natural Fire (2 LFs)
    lf_forest_vegetation_fire,
    lf_strong_forest_fire,

    # ♨️ Industrial Process Heat (2 LFs)
    lf_static_industrial_heat,
    lf_nighttime_process_heat,
]

LABELING_FUNCTION_MAP: Dict[str, Callable[[Any], Optional[str]]] = {
    lf.__name__: lf for lf in ALL_LABELING_FUNCTIONS
}


# ---------------------------------------------------------------------------
# Aggregation & Backward Compatibility Layer
# (Imports consensus logic from app.intelligence.label_aggregator)
# ---------------------------------------------------------------------------
from app.intelligence.label_aggregator import (
    apply_labeling_functions,
    aggregate_votes,
    compute_vote_summary,
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
    "is_missing",
    "get_feature",
    "get_numeric_feature",
    "get_distance_meters",
    "get_frp",
    "get_brightness",
    "get_confidence",
    "is_night",
    "get_firms_type",
]


# ---------------------------------------------------------------------------
# Module Verification & Demonstration Test Harness
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("=" * 76)
    print("THERMOSCOPE-AI: WEAK SUPERVISION LABELING FUNCTIONS TEST HARNESS")
    print("=" * 76)
    print(f"[*] Total Registered Labeling Functions: {len(ALL_LABELING_FUNCTIONS)}")
    for i, lf in enumerate(ALL_LABELING_FUNCTIONS, 1):
        print(f"  {i:02d}. {lf.__name__}")

    print("\n" + "=" * 76)
    print("EXECUTING MANDATORY SPECIFICATION TESTS")
    print("=" * 76)

    test_scenarios = [
        # Test 1: Strong Industrial Context (Industry nearby + High FRP)
        {
            "title": "Test 1: Strong Industrial Context (Industry nearby + High FRP)",
            "record": {
                "latitude": 30.3165,
                "longitude": 78.0322,
                "frp": 65.0,
                "brightness": 345.0,
                "confidence": "high",
                "distance_to_industry_m": 350.0,
                "has_industrial_2km": 1,
            },
            "verify": lambda votes: any(
                votes[k] == INDUSTRIAL_FIRE
                for k in ["lf_industry_high_frp", "lf_factory_proximity_thermal", "lf_industrial_zone_cluster"]
            ),
            "expected_desc": "At least one Industrial LF votes INDUSTRIAL_FIRE",
        },
        # Test 2: Strong Oil/Gas Context (Oil/gas nearby + nighttime evidence)
        {
            "title": "Test 2: Strong Oil/Gas Context (Oil/gas nearby + Night detection)",
            "record": {
                "latitude": 21.6500,
                "longitude": 72.5000,
                "frp": 25.0,
                "brightness": 330.0,
                "is_night": True,
                "distance_to_oil_gas_m": 450.0,
                "distance_to_refinery_m": 1200.0,
            },
            "verify": lambda votes: votes.get("lf_oil_gas_flare") == GAS_FLARE or votes.get("lf_refinery_flare") == GAS_FLARE,
            "expected_desc": "Gas Flare LF votes GAS_FLARE",
        },
        # Test 3: Mining Nearby but Weak Thermal Evidence (FRP=2.0 MW, Brightness=295 K)
        {
            "title": "Test 3: Mining Nearby but Weak Thermal Evidence -> Must ABSTAIN",
            "record": {
                "latitude": 23.8000,
                "longitude": 86.4000,
                "frp": 2.0,
                "brightness": 295.0,
                "confidence": "nominal",
                "distance_to_mining_m": 400.0,
            },
            "verify": lambda votes: votes.get("lf_mining_thermal_activity") is None and votes.get("lf_mining_high_confidence") is None,
            "expected_desc": "Mining LFs ABSTAIN (Proximity alone is insufficient)",
        },
        # Test 4: Agriculture Feature Missing -> All Agriculture LFs MUST ABSTAIN
        {
            "title": "Test 4: Agriculture Feature Missing (None) -> Agriculture LFs Must ABSTAIN",
            "record": {
                "latitude": 30.8000,
                "longitude": 75.8000,
                "frp": 14.0,
                "brightness": 318.0,
                "firms_type": 0,
                "distance_to_agriculture_m": None,  # Missing!
                "distance_to_industry_m": 9200.0,
            },
            "verify": lambda votes: votes.get("lf_agriculture_vegetation_fire") is None and votes.get("lf_agriculture_burn_context") is None,
            "expected_desc": "All Agriculture LFs ABSTAIN due to missing data",
        },
        # Test 5: Forest Feature Missing -> All Forest LFs MUST ABSTAIN
        {
            "title": "Test 5: Forest Feature Missing (None) -> Forest LFs Must ABSTAIN",
            "record": {
                "latitude": 30.5000,
                "longitude": 79.1000,
                "frp": 85.0,
                "brightness": 355.0,
                "firms_type": 0,
                "distance_to_forest_m": None,  # Missing!
                "distance_to_industry_m": 18000.0,
            },
            "verify": lambda votes: votes.get("lf_forest_vegetation_fire") is None and votes.get("lf_strong_forest_fire") is None,
            "expected_desc": "All Forest LFs ABSTAIN due to missing data",
        },
        # Test 6: Valid Agriculture Context (Agriculture nearby + vegetation + stubble FRP)
        {
            "title": "Test 6: Valid Agriculture Context -> Agriculture LF Votes",
            "record": {
                "latitude": 30.8000,
                "longitude": 75.8000,
                "frp": 14.0,
                "brightness": 318.0,
                "firms_type": 0,
                "distance_to_agriculture_m": 250.0,
                "distance_to_industry_m": 9200.0,
            },
            "verify": lambda votes: votes.get("lf_agriculture_vegetation_fire") == AGRICULTURAL_BURN or votes.get("lf_agriculture_burn_context") == AGRICULTURAL_BURN,
            "expected_desc": "Agriculture LF votes AGRICULTURAL_BURN",
        },
        # Test 7: Valid Forest Context (Forest nearby + high FRP + deep isolation from industry)
        {
            "title": "Test 7: Valid Forest Context -> Forest LF Votes",
            "record": {
                "latitude": 30.5000,
                "longitude": 79.1000,
                "frp": 85.0,
                "brightness": 355.0,
                "firms_type": 0,
                "distance_to_forest_m": 400.0,
                "distance_to_industry_m": 18000.0,
            },
            "verify": lambda votes: votes.get("lf_forest_vegetation_fire") == FOREST_NATURAL_FIRE or votes.get("lf_strong_forest_fire") == FOREST_NATURAL_FIRE,
            "expected_desc": "Forest LF votes FOREST_NATURAL_FIRE",
        },
        # Test 8: Process Heat Context (Static source type 3 + industry proximity)
        {
            "title": "Test 8: Process Heat Context -> Process Heat LF Votes",
            "record": {
                "latitude": 22.8000,
                "longitude": 86.2000,
                "frp": 11.0,
                "brightness": 315.0,
                "firms_type": 3,
                "is_night": True,
                "distance_to_industry_m": 450.0,
            },
            "verify": lambda votes: votes.get("lf_static_industrial_heat") == INDUSTRIAL_PROCESS_HEAT or votes.get("lf_nighttime_process_heat") == INDUSTRIAL_PROCESS_HEAT,
            "expected_desc": "Process Heat LF votes INDUSTRIAL_PROCESS_HEAT",
        },
        # Test 9: Missing / NaN / String / Corrupt Values -> No Crash & Safe ABSTAIN
        {
            "title": "Test 9: Corrupt / NaN / Type-mismatched Features -> Safe ABSTAIN (No crash)",
            "record": {
                "latitude": "invalid_lat",
                "longitude": None,
                "frp": float("nan"),
                "brightness": "unknown",
                "distance_to_industry_m": 999.0,  # Sentinel missing
                "distance_to_refinery_m": float("inf"),
            },
            "verify": lambda votes: all(v is None for v in votes.values()),
            "expected_desc": "All LFs safely ABSTAIN without exceptions",
        },
    ]

    all_passed = True
    for test in test_scenarios:
        votes = apply_labeling_functions(test["record"])
        passed = test["verify"](votes)
        consensus = aggregate_votes(votes)
        active_votes = [f"{k} -> {v}" for k, v in votes.items() if v is not None]

        if not passed:
            all_passed = False

        status = "[PASS]" if passed else "[FAIL]"
        print(f"\n{status} | {test['title']}")
        print(f"       Requirement: {test['expected_desc']}")
        print(f"       Consensus  : {consensus}")
        print(f"       Active LFs : {len(active_votes)} voted ({', '.join(active_votes) if active_votes else 'All Abstained'})")

    print("\n" + "=" * 76)
    if all_passed:
        print("[*] ALL TEST SCENARIOS PASSED SUCCESSFULLY (100% SPEC COMPLIANCE)!")
    else:
        print("[!] SOME TEST SCENARIOS FAILED!")
    print("=" * 76)
