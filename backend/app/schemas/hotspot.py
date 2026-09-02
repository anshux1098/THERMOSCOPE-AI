"""
hotspot.py
Pydantic schema definitions for thermal hotspots detected by NASA FIRMS.
"""
from typing import Any, Dict, Optional, Union
from pydantic import BaseModel, Field


def normalize_confidence(conf: Any) -> str:
    """
    Normalize NASA FIRMS confidence representation to standard string labels.
    - VIIRS returns: 'h' (high), 'n' (nominal), 'l' (low)
    - MODIS returns: integer 0-100 or string percentage
    """
    if conf is None:
        return "nominal"
    
    val = str(conf).strip().lower()
    if val in ("h", "high"):
        return "high"
    elif val in ("n", "nominal", "med", "medium"):
        return "nominal"
    elif val in ("l", "low"):
        return "low"
    
    # Try parsing as numeric percentage (MODIS format)
    try:
        score = float(val)
        if score >= 80:
            return "high"
        elif score >= 30:
            return "nominal"
        else:
            return "low"
    except ValueError:
        return val


class HotspotBase(BaseModel):
    latitude: float = Field(..., description="Latitude coordinate in decimal degrees")
    longitude: float = Field(..., description="Longitude coordinate in decimal degrees")
    frp: float = Field(..., description="Fire Radiative Power in Megawatts (MW)")
    brightness: float = Field(..., description="Brightness temperature in Kelvin")
    confidence: str = Field(..., description="Detection confidence: 'high', 'nominal', or 'low'")
    acq_date: str = Field(..., description="Acquisition date in YYYY-MM-DD format")


class Hotspot(HotspotBase):
    """
    Thermal Hotspot schema matching the project MVP output structure.
    """
    model_config = {
        "json_schema_extra": {
            "example": {
                "latitude": 30.3165,
                "longitude": 78.0322,
                "frp": 42.5,
                "brightness": 325.4,
                "confidence": "high",
                "acq_date": "2026-09-01"
            }
        }
    }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Hotspot":
        """
        Create a standardized Hotspot instance from raw dictionary or FIRMS record.
        Handles both VIIRS ('bright_ti4') and MODIS ('brightness') column names.
        """
        lat = float(data.get("latitude", 0.0))
        lon = float(data.get("longitude", 0.0))
        frp = float(data.get("frp", 0.0))
        
        # Determine brightness temperature from available columns
        brightness_val = data.get("brightness")
        if brightness_val is None:
            brightness_val = data.get("bright_ti4")
        if brightness_val is None:
            brightness_val = data.get("bright_ti5")
        if brightness_val is None:
            brightness_val = data.get("bright_t31", 0.0)
        brightness = float(brightness_val)

        conf = normalize_confidence(data.get("confidence"))
        acq_date = str(data.get("acq_date", "")).strip()

        return cls(
            latitude=lat,
            longitude=lon,
            frp=round(frp, 2),
            brightness=round(brightness, 2),
            confidence=conf,
            acq_date=acq_date
        )


def row_to_hotspot(row: Union[Dict[str, Any], Any]) -> Hotspot:
    """Helper to convert a pandas Series or dictionary to Hotspot."""
    if hasattr(row, "to_dict"):
        row = row.to_dict()
    return Hotspot.from_dict(row)
