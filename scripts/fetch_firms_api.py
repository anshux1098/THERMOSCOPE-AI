"""
fetch_firms_api.py — FIXED: proper .env reading + direct token support
"""
import json
import os
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

# Fix pathlib: keep as Path objects
# scripts/fetch_firms_api.py → parents[0]=scripts, parents[1]=project root
# .env lives at project root (parents[1]), not in scripts/
_script_dir = Path(__file__).resolve().parent          # scripts/
_root_dir  = _script_dir.parent                          # project root
backend_dir = _root_dir / "backend"                      # for sys.path
root_dir = _root_dir                                     # for ENV_PATH
for p in (str(backend_dir), str(root_dir)):
    if p not in sys.path:
        sys.path.insert(0, p)

import pandas as pd
import requests

ENV_PATH = root_dir / ".env"   # project root/.env (correct)
INDIA_BBOX = "68,-10,96,40"
FIRMS_API_URLS = [
    "https://firmsmodapi.azure-api.net/api/v1/fireweb/firearch",
    "https://firmsmodapi.azure-api.net/api/v1/fireweb/firearchbulk",
]


def _get_nested(obj, path):
    for k in path.split("/"):
        if isinstance(obj, dict) and k in obj:
            obj = obj[k]
        else:
            return None
    return obj


def _parse_firms_json(fires):
    rows = []
    for f in fires:
        row = {}
        for key, dest, cast in [
            ("latitude", "latitude", float),
            ("lat", "latitude", float),
            ("longitude", "longitude", float),
            ("lon", "longitude", float),
            ("frp", "frp", float),
            ("FRPF", "frp", float),
            ("bright_ti4", "bright_ti4", float),
            ("Brightness_TI4", "bright_ti4", float),
            ("brightness", "bright_ti4", float),
            ("bright_ti5", "bright_ti5", float),
            ("Brightness_TI5", "bright_ti5", float),
            ("acq_date", "acq_date", str),
            ("acquisitionDate", "acq_date", str),
            ("acq_time", "acq_time", str),
            ("acquisitionTime", "acq_time", str),
            ("satellite", "satellite", str),
            ("satelliteShortName", "satellite", str),
            ("confidence", "confidence", int),
            ("confidenceLevel", "confidence", int),
            ("daynight", "daynight", str),
            ("daytimeNighttimeFlag", "daynight", str),
        ]:
            val = _get_nested(f, key)
            if val is not None:
                try:
                    row[dest] = cast(val)
                    if dest == "acq_date":
                        row[dest] = str(val)[:10]
                    if dest == "daynight" and val:
                        row[dest] = str(val)[:1].upper()
                    break
                except (ValueError, TypeError):
                    pass
        row["source_dataset"] = _get_nested(f, "datasetname") or _get_nested(f, "dataset") or "FIRMS_API"
        if "latitude" in row and "longitude" in row:
            rows.append(row)
    return pd.DataFrame(rows)


def fetch_firms_data(token, start_date, end_date, datasets="VIIRS_SNPP_NRT,VIIRS_NOAA20_NRT"):
    params = {"format": "json", "startdate": start_date, "enddate": end_date, "bbox": INDIA_BBOX}
    if datasets:
        params["datasetnames"] = datasets
    headers = {"Authorization": f"Bearer {token}"}
    for url in FIRMS_API_URLS:
        try:
            r = requests.get(url, params=params, headers=headers, timeout=45)
            print(f"  {url.split('//')[1]}: {r.status_code}")
            if r.status_code == 200:
                data = r.json()
                if isinstance(data, list):
                    return _parse_firms_json(data)
                elif isinstance(data, dict):
                    for key in ("fires", "data", "features"):
                        if key in data:
                            return _parse_firms_json(data[key])
                    return _parse_firms_json([data])
                return pd.DataFrame()
            elif r.status_code == 401:
                print("    401 Unauthorized — token may be invalid")
                return None
            elif r.status_code == 429:
                time.sleep(5)
                continue
            else:
                body = r.text[:200].replace("\n", " ")
                print(f"    {r.status_code}: {body}")
        except Exception as e:
            print(f"    Error: {type(e).__name__}: {str(e)[:100]}")
        time.sleep(1)
    return None


def main():
    print("=" * 60)
    print("FIRMS API — Fetch Additional Hotspot Data")
    print("=" * 60)

    from app.core.config import get_settings
    settings = get_settings()
    token = settings.FIRMS_MAP_KEY or os.environ.get("FIRMS_MAP_KEY", "")

    if not token:
        print("\nERROR: FIRMS_MAP_KEY not found.")
        print("Manual fix — run this in PowerShell first:")
        print('  $env:FIRMS_MAP_KEY = "your_actual_token_here"')
        print("Then run: python -m scripts.fetch_firms_api")
        print("\nOr add to .env:")
        print('  FIRMS_MAP_KEY=your_token_here')
        sys.exit(1)

    print(f"Token: {'*' * 6}{token[-4:] if len(token) > 4 else token}")
    days = int(os.environ.get("FIRMS_DAYS", "30"))
    datasets = os.environ.get("FIRMS_DATASETS", "VIIRS_SNPP_NRT,VIIRS_NOAA20_NRT")
    print(f"Days back: {days} | Datasets: {datasets}")
    print(f"India bbox: {INDIA_BBOX}")
    print()

    existing_path = root_dir / "data/raw/firms_recent.csv"
    if existing_path.exists():
        existing = pd.read_csv(existing_path, low_memory=False)
        print(f"Existing: {len(existing)} hotspots, dates: {existing['acq_date'].min()} → {existing['acq_date'].max()}")
    else:
        existing = pd.DataFrame()
        print("No existing firms_recent.csv — fresh start")

    end_date = datetime.now().strftime("%Y-%m-%d")
    start_date = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")
    print(f"\nFetching: {start_date} → {end_date}")
    print("-" * 40)
    t0 = time.time()
    new_df = fetch_firms_data(token, start_date, end_date, datasets)
    elapsed = time.time() - t0

    if new_df is None:
        print(f"\nFAILED after {elapsed:.0f}s")
        sys.exit(1)
    if new_df.empty:
        print(f"\nNo hotspots returned (empty response) after {elapsed:.0f}s")
        print("Will use existing data only.")
        existing.to_csv(existing_path, index=False)
        print(f"Existing saved: {len(existing)} hotspots")
        sys.exit(0)

    print(f"\nGot {len(new_df)} new hotspots in {elapsed:.0f}s")
    print(f"  Date range: {new_df['acq_date'].min()} → {new_df['acq_date'].max()}")
    if "satellite" in new_df.columns:
        print(f"  By satellite: {dict(new_df['satellite'].value_counts())}")

    expected_cols = [
        "latitude", "longitude", "bright_ti4", "scan", "track",
        "acq_date", "acq_time", "satellite", "instrument", "confidence",
        "version", "bright_ti5", "frp", "daynight", "source_dataset",
    ]

    # Standardize
    for col in expected_cols:
        if col not in new_df.columns:
            new_df[col] = None

    new_df = new_df[expected_cols]
    combined = pd.concat([existing, new_df], ignore_index=True)

    before = len(combined)
    combined = combined.drop_duplicates(subset=["latitude", "longitude"], keep="first", ignore_index=True)
    after = len(combined)
    dupes = before - after

    print(f"\nCombined: {before} → {after} (removed {dupes} duplicates)")

    os.makedirs(existing_path.parent, exist_ok=True)
    combined.to_csv(existing_path, index=False)
    print(f"\nSaved → {existing_path}")
    print(f"  Total hotspots: {after}")
    print(f"  New added: {len(new_df)}")
    print(f"  Existing kept: {len(existing)}")
    print(f"  Duplicates removed: {dupes}")

    # Quick stats
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Total hotspots: {len(combined)}")
    print(f"Date range: {combined['acq_date'].min()} → {combined['acq_date'].max()}")
    if "satellite" in combined.columns:
        print(f"By satellite: {dict(combined['satellite'].value_counts())}")
    if "frp" in combined.columns:
        print(f"FRP range: {combined['frp'].min():.1f} – {combined['frp'].max():.1f} (mean {combined['frp'].mean():.1f})")


if __name__ == "__main__":
    main()
