"""
firms_fetcher.py
Fetch NASA FIRMS active fire/hotspot data for a region and date range.
Owner: P1 - Data Engineer
"""
import os
import time
import pandas as pd
import requests
from io import StringIO
from app.core.config import get_settings

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


def save_firms_data(df, path="data/raw/firms_recent.csv"):
    """Save FIRMS DataFrame to CSV for offline use."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False)
    print(f"Saved {len(df)} hotspots to {path}")


def load_firms_data(path="data/raw/firms_recent.csv"):
    """Load cached FIRMS data if it exists."""
    if os.path.exists(path):
        df = pd.read_csv(path)
        print(f"Loaded {len(df)} cached hotspots from {path}")
        return df
    return None


def fetch_firms_data(dataset, days=7, bbox=INDIA_BBOX):
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


def fetch_multiple_datasets(days=7, datasets=None, bbox=INDIA_BBOX, cache_path="data/raw/firms_recent.csv"):
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
        return pd.DataFrame()
    combined = pd.concat(frames, ignore_index=True)
    print(f"\nTotal combined records: {len(combined)}")
    if cache_path:
        save_firms_data(combined, cache_path)
    return combined


def get_data_availability():
    if not MAP_KEY or MAP_KEY == "your_map_key_here":
        raise ValueError("FIRMS_MAP_KEY not set.")
    url = f"https://firms.modaps.eosdis.nasa.gov/api/data_availability/csv/{MAP_KEY}/all"
    response = requests.get(url, timeout=30)
    response.raise_for_status()
    return pd.read_csv(StringIO(response.text))


if __name__ == "__main__":
    print("=" * 60)
    print("FIRMS Data Fetcher - Test Run")
    print("=" * 60)
    try:
        avail = get_data_availability()
        print("\n[1] Data availability:")
        print(avail[["data_id", "min_date", "max_date"]].to_string(index=False))
    except Exception as e:
        print(f"  Skipped: {e}")
    try:
        df = fetch_multiple_datasets(days=3, datasets=["VIIRS_SNPP_NRT"])
        print(f"\n[2] Fetched {len(df)} VIIRS hotspots")
        print(f"  Columns: {df.columns.tolist()}")
        print(f"  Confidence: {df['confidence'].value_counts().to_dict()}")
        print(f"  Day/Night: {df['daynight'].value_counts().to_dict()}")
        print(f"  FRP range: {df['frp'].min():.1f} - {df['frp'].max():.1f} MW")
    except Exception as e:
        print(f"  Error: {e}")
