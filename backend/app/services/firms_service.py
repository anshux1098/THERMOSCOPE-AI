"""
firms_service.py
Fetch NASA FIRMS active fire/hotspot data for a region and date range,
standardized with the Hotspot Pydantic schema.
"""
import os
import sys
import json
import time
from pathlib import Path
from typing import List, Optional
from io import StringIO
import pandas as pd
import requests

# Ensure app package is discoverable
backend_dir = str(Path(__file__).resolve().parents[2])
root_dir = str(Path(__file__).resolve().parents[3])
for p in (backend_dir, root_dir):
    if p not in sys.path:
        sys.path.insert(0, p)

from app.core.config import get_settings
from app.schemas.hotspot import Hotspot, row_to_hotspot

settings = get_settings()

MAP_KEY = settings.FIRMS_MAP_KEY

DATASETS = {
    "VIIRS_SNPP_NRT": "Suomi NPP VIIRS (375m, 2x/day)",
    "VIIRS_NOAA20_NRT": "NOAA-20 VIIRS (375m, 2x/day)",
    "VIIRS_NOAA21_NRT": "NOAA-21 VIIRS (375m, 2x/day)",
    "MODIS_NRT": "MODIS Terra/Aqua (1km, 4x/day)",
}

INDIA_BBOX = settings.INDIA_BBOX
FIRMS_BASE_URL = "https://firms.modaps.eosdis.nasa.gov/api/area/csv"


def save_firms_data(df: pd.DataFrame, path: str = "data/raw/firms_recent.csv"):
    """Save FIRMS DataFrame to CSV for offline use."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False)
    print(f"Saved {len(df)} hotspots to {path}")


def load_firms_data(path: str = "data/raw/firms_recent.csv") -> Optional[pd.DataFrame]:
    """Load cached FIRMS data if it exists."""
    if os.path.exists(path):
        df = pd.read_csv(path)
        print(f"Loaded {len(df)} cached hotspots from {path}")
        return df
    return None


def fetch_firms_data(dataset: str, days: int = 7, bbox = INDIA_BBOX) -> pd.DataFrame:
    if not MAP_KEY or MAP_KEY == "your_map_key_here":
        raise ValueError("FIRMS_MAP_KEY not set. Get free key at firms.modaps.eosdis.nasa.gov/api/map_key")
    if dataset not in DATASETS:
        raise ValueError(f"Invalid dataset. Choose from: {list(DATASETS.keys())}")
    if days < 1 or days > 10:
        raise ValueError("Days must be between 1 and 10 for NRT data")

    url = f"{FIRMS_BASE_URL}/{MAP_KEY}/{dataset}/{bbox[0]},{bbox[1]},{bbox[2]},{bbox[3]}/{days}"
    print(f"Fetching {dataset} data for last {days} days...")
    response = requests.get(url, timeout=60)
    response.raise_for_status()
    df = pd.read_csv(StringIO(response.text))
    print(f"  Received {len(df)} hotspot records")
    return df


def fetch_multiple_datasets(days: int = 7, datasets: Optional[List[str]] = None, bbox = INDIA_BBOX, cache_path: str = "data/raw/firms_recent.csv") -> pd.DataFrame:
    if cache_path and os.path.exists(cache_path):
        cache_age = time.time() - os.path.getmtime(cache_path)
        if cache_age < 86400:  # Less than 24 hours old
            df = load_firms_data(cache_path)
            if df is not None and not df.empty:
                return df

    if datasets is None:
        datasets = ["VIIRS_SNPP_NRT", "VIIRS_NOAA20_NRT"]
    frames = []
    for ds in datasets:
        try:
            df = fetch_firms_data(ds, days, bbox)
            df["source_dataset"] = ds
            frames.append(df)
        except Exception as e:
            print(f"  Warning: Failed to fetch {ds}: {e}")
            continue
    if not frames:
        # Fallback to load cache even if older than 24 hours
        if cache_path and os.path.exists(cache_path):
            df = load_firms_data(cache_path)
            if df is not None and not df.empty:
                return df
        return pd.DataFrame()
    combined = pd.concat(frames, ignore_index=True)
    print(f"\nTotal combined records: {len(combined)}")
    if cache_path:
        save_firms_data(combined, cache_path)
    return combined


def get_data_availability() -> pd.DataFrame:
    if not MAP_KEY or MAP_KEY == "your_map_key_here":
        raise ValueError("FIRMS_MAP_KEY not set.")
    url = f"https://firms.modaps.eosdis.nasa.gov/api/data_availability/csv/{MAP_KEY}/all"
    response = requests.get(url, timeout=30)
    response.raise_for_status()
    return pd.read_csv(StringIO(response.text))


def parse_hotspots_from_df(df: pd.DataFrame) -> List[Hotspot]:
    """
    Convert a FIRMS DataFrame into a list of standardized Hotspot schema models.
    """
    if df is None or df.empty:
        return []
    
    hotspots = []
    for _, row in df.iterrows():
        try:
            hotspots.append(row_to_hotspot(row))
        except Exception:
            continue
    return hotspots


def get_standardized_hotspots(days: int = 7, datasets: Optional[List[str]] = None, bbox = INDIA_BBOX, cache_path: str = "data/raw/firms_recent.csv") -> List[Hotspot]:
    """
    Fetch/load FIRMS data and return validated Hotspot schema models.
    """
    df = fetch_multiple_datasets(days=days, datasets=datasets, bbox=bbox, cache_path=cache_path)
    return parse_hotspots_from_df(df)


def get_hotspot_dicts(days: int = 7, datasets: Optional[List[str]] = None, bbox = INDIA_BBOX, cache_path: str = "data/raw/firms_recent.csv") -> List[dict]:
    """
    Fetch/load FIRMS data and return list of dictionaries in exact Hotspot schema format.
    """
    hotspots = get_standardized_hotspots(days=days, datasets=datasets, bbox=bbox, cache_path=cache_path)
    return [h.model_dump() for h in hotspots]


if __name__ == "__main__":
    print("=" * 60)
    print("FIRMS Service & Hotspot Schema Integration - Test Run")
    print("=" * 60)
    
    # 1. Check Data Availability if MAP_KEY is set
    try:
        avail = get_data_availability()
        print("\n[1] Data availability:")
        print(avail[["data_id", "min_date", "max_date"]].to_string(index=False))
    except Exception as e:
        print(f"\n[1] Data availability check skipped: {e}")

    # 2. Fetch and Standardize Hotspots with Schema
    print("\n[2] Fetching and parsing hotspots into Hotspot schema...")
    hotspots = get_standardized_hotspots(days=3, datasets=["VIIRS_SNPP_NRT"])
    print(f"  Successfully parsed {len(hotspots)} hotspots into Hotspot schema.")

    if hotspots:
        print("\n[3] Sample Formatted Hotspot Schema Output (first 3):")
        for i, h in enumerate(hotspots[:3], 1):
            print(f"\n--- Hotspot #{i} ---")
            print(json.dumps(h.model_dump(), indent=2))
