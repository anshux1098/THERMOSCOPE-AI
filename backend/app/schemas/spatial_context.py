"""
spatial_context.py
Pydantic schema model for Spatial Context distances.
"""
from typing import Optional, Union
from pydantic import BaseModel, Field


class SpatialContext(BaseModel):
    """
    Geospatial distances (in meters) from the hotspot to nearest land-use features:

    Industrial:
    - Industry        (general factories / manufacturing / industrial zones)
    - Refinery        (petroleum / chemical refineries)
    - Oil & Gas       (extraction / processing facilities)
    - Mining          (quarries / coal / mineral extraction)

    Natural:
    - Forest          (woodlands / scrub)
    - Agriculture     (farmland / crops / orchards)

    Infrastructure:
    - Power Plant     (power generation stations)
    """
    # 🏭 Industrial
    nearest_industry_m: Optional[Union[int, float]] = Field(
        None, description="Distance to nearest industrial site / factory in meters"
    )

    # 🔥 Oil/Gas/Persistent thermal sources
    nearest_refinery_m: Optional[Union[int, float]] = Field(
        None, description="Distance to nearest petroleum or chemical refinery in meters"
    )
    nearest_oil_gas_m: Optional[Union[int, float]] = Field(
        None, description="Distance to nearest oil & gas extraction or processing facility in meters"
    )

    # ⛏️ Mining
    nearest_mining_m: Optional[Union[int, float]] = Field(
        None, description="Distance to nearest mining site / quarry in meters"
    )

    # 🌾 Agriculture
    nearest_agriculture_m: Optional[Union[int, float]] = Field(
        None, description="Distance to nearest agricultural land in meters"
    )

    # 🌲 Forest
    nearest_forest_m: Optional[Union[int, float]] = Field(
        None, description="Distance to nearest forest in meters"
    )

    # ⚡ Additional infrastructure
    nearest_power_plant_m: Optional[Union[int, float]] = Field(
        None, description="Distance to nearest power plant in meters"
    )

    model_config = {
        "json_schema_extra": {
            "example": {
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
