"""
analysis.py
Combined analysis schema uniting Hotspot and SpatialContext.
Provides dynamic analysis calculation for any real NASA FIRMS hotspot.
"""
import sys
import json
from pathlib import Path
from typing import Any, Dict, Optional, Union
from pydantic import BaseModel

# Ensure app package is discoverable
backend_dir = str(Path(__file__).resolve().parents[2])
root_dir = str(Path(__file__).resolve().parents[3])
for p in (backend_dir, root_dir):
    if p not in sys.path:
        sys.path.insert(0, p)

from app.schemas.hotspot import Hotspot
from app.schemas.spatial_context import SpatialContext


class HotspotAnalysis(BaseModel):
    """
    Combined Hotspot Analysis output model (Phase D):
    - hotspot: Core NASA FIRMS thermal attributes
    - spatial_context: Proximity distances in meters for 7 categories:
      Industry, Refinery, Oil & Gas, Mining, Agriculture, Forest, Power Plant
    - classification: Hybrid Intelligence decision (14 LFs + XGBoost fusion)
    """
    hotspot: Hotspot
    spatial_context: SpatialContext
    classification: Optional[Dict[str, Any]] = None

    # OpenAPI documentation example (used by FastAPI/Swagger UI documentation only)
    model_config = {
        "json_schema_extra": {
            "example": {
                "hotspot": {
                    "latitude": 30.3165,
                    "longitude": 78.0322,
                    "frp": 42.5,
                    "brightness": 325.4,
                    "confidence": "high",
                    "acq_date": "2026-09-01"
                },
                "spatial_context": {
                    "nearest_industry_m": 450,
                    "nearest_refinery_m": 12400,
                    "nearest_oil_gas_m": 8750,
                    "nearest_mining_m": 22300,
                    "nearest_agriculture_m": 850,
                    "nearest_forest_m": 3200,
                    "nearest_power_plant_m": 5100,
                }
            }
        }
    }

    @classmethod
    def from_hotspot(
        cls,
        hotspot_data: Union[Hotspot, Dict[str, Any]],
        radius_meters: int = 15000,
        use_live_api: bool = False
    ) -> "HotspotAnalysis":
        """
        Dynamically analyzes ANY real hotspot:
        1. Takes NASA FIRMS hotspot coordinates
        2. Queries OSM for surrounding features (industry, forest, agriculture)
        3. Calculates real distances in meters using Distance Engine
        4. Populates and returns a real HotspotAnalysis instance
        """
        from app.geo.spatial_context import compute_geospatial_context

        if isinstance(hotspot_data, dict):
            hotspot_model = Hotspot.from_dict(hotspot_data)
        elif isinstance(hotspot_data, Hotspot):
            hotspot_model = hotspot_data
        else:
            raise ValueError(f"Invalid hotspot data: {type(hotspot_data)}")

        # Calculate live spatial context using OSM + Distance Engine
        context_result = compute_geospatial_context(
            hotspot=hotspot_model,
            radius_meters=radius_meters,
            use_live_api=use_live_api
        )
        summary = context_result.get("summary_distances", {})

        ind_m       = summary.get("distance_to_industry_m")
        ref_m       = summary.get("distance_to_refinery_m")
        oil_gas_m   = summary.get("distance_to_oil_gas_m")
        mining_m    = summary.get("distance_to_mining_m")
        agri_m      = summary.get("distance_to_agriculture_m")
        forest_m    = summary.get("distance_to_forest_m")
        power_m     = summary.get("distance_to_power_plant_m")

        spatial_ctx = SpatialContext(
            nearest_industry_m=int(round(ind_m))       if ind_m     is not None else None,
            nearest_refinery_m=int(round(ref_m))       if ref_m     is not None else None,
            nearest_oil_gas_m=int(round(oil_gas_m))    if oil_gas_m is not None else None,
            nearest_mining_m=int(round(mining_m))      if mining_m  is not None else None,
            nearest_agriculture_m=int(round(agri_m))   if agri_m    is not None else None,
            nearest_forest_m=int(round(forest_m))      if forest_m  is not None else None,
            nearest_power_plant_m=int(round(power_m))  if power_m   is not None else None,
        )

        return cls(hotspot=hotspot_model, spatial_context=spatial_ctx)


if __name__ == "__main__":
    print("=" * 65)
    print("ANALYZING REAL NASA FIRMS HOTSPOT (DYNAMIC CALCULATION)")
    print("=" * 65)

    from app.services.firms_service import get_standardized_hotspots

    # 1. Fetch real hotspots from NASA FIRMS data
    real_hotspots = get_standardized_hotspots(days=3)
    if real_hotspots:
        selected_hotspot = real_hotspots[0]
        print(f"\n[+] Selected Real Hotspot from FIRMS:")
        print(f"    Location: ({selected_hotspot.latitude}, {selected_hotspot.longitude})")
        print(f"    FRP: {selected_hotspot.frp} MW | Brightness: {selected_hotspot.brightness} K")
        print(f"    Confidence: {selected_hotspot.confidence} | Date: {selected_hotspot.acq_date}")
    else:
        # Fallback coordinate if no cached FIRMS data
        selected_hotspot = Hotspot(
            latitude=28.7786,
            longitude=77.1428,
            frp=25.4,
            brightness=330.2,
            confidence="high",
            acq_date="2026-08-30"
        )

    # 2. Dynamically calculate spatial context for this specific hotspot
    analysis = HotspotAnalysis.from_hotspot(selected_hotspot, radius_meters=15000, use_live_api=False)

    print("\n[+] Dynamic Output for this specific Hotspot:")
    print(json.dumps(analysis.model_dump(), indent=2))
    print("=" * 65)
