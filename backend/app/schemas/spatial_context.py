"""
spatial_context.py
Pydantic schema model for Spatial Context distances.
"""
from typing import Optional, Union
from pydantic import BaseModel, Field


class SpatialContext(BaseModel):
    """
    Geospatial distances (in meters) from the hotspot to nearest land-use features:
    - Industry
    - Forest
    - Agriculture
    """
    nearest_industry_m: Optional[Union[int, float]] = Field(
        None, description="Distance to nearest industrial site / factory in meters"
    )
    nearest_forest_m: Optional[Union[int, float]] = Field(
        None, description="Distance to nearest forest in meters"
    )
    nearest_agriculture_m: Optional[Union[int, float]] = Field(
        None, description="Distance to nearest agricultural land in meters"
    )

    model_config = {
        "json_schema_extra": {
            "example": {
                "nearest_industry_m": 450,
                "nearest_forest_m": 3200,
                "nearest_agriculture_m": 850
            }
        }
    }
