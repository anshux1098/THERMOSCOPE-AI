"""
hotspot.py
Pydantic schema definitions for thermal hotspots detected by NASA FIRMS.
"""
from typing import Any, Dict, Optional, Union
from pydantic import BaseModel, Field


def _is_nullish(val: Any) -> bool:
    """True for None, NaN, or empty string — safe against pandas/float NaN inputs."""
    if val is None:
        return True
    if isinstance(val, str):
        return val.strip() == ""
    try:
        f = float(val)
        return f != f  # NaN
    except (TypeError, ValueError):
        return False


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
    bright_ti5: Optional[float] = Field(
        None,
        description=(
            "Optional VIIRS 5um brightness temperature (Kelvin). Official production "
            "model feature. When omitted the ML feature vector defaults bright_ti5 to 0.0, "
            "which does NOT represent parity with a batch record carrying the real value."
        ),
    )
    daynight: Optional[str] = Field(
        None,
        description=(
            "Optional VIIRS day/night detection flag ('D' or 'N'). Night-gated labeling "
            "functions (e.g., refinery/oil-and-gas flare) abstain when omitted, which "
            "breaks batch/live parity. Optional; legacy clients that cannot supply it get "
            "the previous (daytime-assumed) behavior."
        ),
    )
    satellite: Optional[str] = Field(
        None,
        description="Optional VIIRS satellite name (e.g., 'N', 'N20'). Contract-only passthrough.",
    )
    acq_time: Optional[str] = Field(
        None,
        description="Optional VIIRS acquisition time in HHMM form. Contract-only passthrough.",
    )


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

        bright_ti5 = data.get("bright_ti5")
        if bright_ti5 is not None and not _is_nullish(bright_ti5):
            bright_ti5 = float(bright_ti5)
        else:
            bright_ti5 = None

        daynight = data.get("daynight")
        if daynight is None or _is_nullish(daynight):
            daynight = None
        else:
            daynight = str(daynight).strip().upper()

        satellite = data.get("satellite")
        if satellite is None or _is_nullish(satellite):
            satellite = None
        else:
            satellite = str(satellite).strip()

        acq_time = data.get("acq_time")
        if acq_time is None or _is_nullish(acq_time):
            acq_time = None
        else:
            acq_time = str(acq_time).strip()

        return cls(
            latitude=lat,
            longitude=lon,
            frp=round(frp, 2),
            brightness=round(brightness, 2),
            confidence=conf,
            acq_date=acq_date,
            bright_ti5=bright_ti5,
            daynight=daynight,
            satellite=satellite,
            acq_time=acq_time,
        )


def row_to_hotspot(row: Union[Dict[str, Any], Any]) -> Hotspot:
    """Helper to convert a pandas Series or dictionary to Hotspot."""
    if hasattr(row, "to_dict"):
        row = row.to_dict()
    return Hotspot.from_dict(row)
