"""
build_demo_dataset.py  [RENAMED FROM build_synth_v2.py]

⚠️  FOR UI DEMO / OFFLINE FALLBACK DISPLAY ONLY.
    NEVER FEED OUTPUT INTO app/ml/train.py OR app/ml/dataset_builder.py.

This script generates a SYNTHETIC, balanced multi-class dataset by sampling
feature values from label-dependent ranges. The labels and features are
fabricated from scratch — they do NOT reflect real NASA FIRMS observations
or real OSM spatial distances.

Use this ONLY for:
  - Offline UI demonstrations
  - Testing the visualization pipeline without internet access
  - Sanity-checking label schema and class names

For real model training, use: scripts/build_real_dataset.py
"""
from __future__ import annotations

import os
import sys
from pathlib import Path
import random
import numpy as np
import pandas as pd

if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

# Make app package importable
backend_dir = str(Path(__file__).resolve().parents[1] / "backend")
root_dir = str(Path(__file__).resolve().parents[1])
for p in (backend_dir, root_dir):
    if p not in sys.path:
        sys.path.insert(0, p)

from app.core.constants import CLASS_LABELS
from app.core.paths import FIRMS_DATASET_PATH, CLASSIFIED_DATASET_PATH

FIRMS_CSV = str(FIRMS_DATASET_PATH)
OUTPUT_CSV = str(CLASSIFIED_DATASET_PATH)
MISSING_KM = 999.0
MISSING_M = 45000.0


def build_balanced_dataset(random_seed: int = 42):
    if not os.path.exists(FIRMS_CSV):
        raise FileNotFoundError(f"FIRMS CSV not found: {FIRMS_CSV}")

    print(f"[build_synth_v2] Loading {FIRMS_CSV}...")
    firms = pd.read_csv(FIRMS_CSV)
    n_rows = len(firms)
    print(f"  -> Total input FIRMS hotspots: {n_rows}")

    np.random.seed(random_seed)
    random.seed(random_seed)

    # Assign archetypes across the dataset to reflect balanced real-world diversity:
    # 0: Industrial Fire (~18%)
    # 1: Gas Flare (~16%)
    # 2: Mining Activity (~14%)
    # 3: Agricultural Burn (~20%)
    # 4: Forest / Natural Fire (~18%)
    # 5: Industrial Process Heat (~10%)
    # 6: Unclassified (~4%)
    archetype_probs = [0.18, 0.16, 0.14, 0.20, 0.18, 0.10, 0.04]
    archetypes = np.random.choice(7, size=n_rows, p=archetype_probs)

    rows = []
    for idx, (_, row) in enumerate(firms.iterrows()):
        hot_lat = float(row.get("latitude", 22.0))
        hot_lon = float(row.get("longitude", 78.0))
        arch = archetypes[idx]

        # Base FIRMS attributes
        frp = float(row.get("frp", 20.0))
        bright_ti4 = float(row.get("bright_ti4", 325.0))
        bright_ti5 = float(row.get("bright_ti5", 295.0))
        confidence = str(row.get("confidence", "nominal")).strip().lower()
        daynight = str(row.get("daynight", "D")).strip().upper()

        # Defaults for distances (in meters and km)
        dist_ind_m = float(np.random.uniform(5000, 25000))
        dist_ref_m = float(np.random.uniform(8000, 35000))
        dist_oil_m = float(np.random.uniform(8000, 35000))
        dist_min_m = float(np.random.uniform(10000, 40000))
        dist_agr_m = float(np.random.uniform(5000, 25000))
        dist_for_m = float(np.random.uniform(8000, 35000))
        dist_pow_m = float(np.random.uniform(10000, 40000))

        firms_type = -1
        is_night_val = daynight == "N"
        persistence = 0.0

        if arch == 0:
            # 🏭 Industrial Fire: close to industry (<1000m), high FRP, high brightness
            dist_ind_m = float(np.random.uniform(150, 850))
            dist_ref_m = float(np.random.uniform(3500, 15000))
            dist_agr_m = float(np.random.uniform(3000, 12000))
            dist_for_m = float(np.random.uniform(6000, 20000))
            frp = float(np.random.uniform(38.0, 95.0))
            bright_ti4 = float(np.random.uniform(335.0, 365.0))
            confidence = "high"

        elif arch == 1:
            # 🛢️ Gas Flare: close to refinery or oil/gas (<1800m), night or persistent
            if random.random() < 0.6:
                dist_ref_m = float(np.random.uniform(250, 1500))
                dist_oil_m = float(np.random.uniform(1200, 4000))
            else:
                dist_oil_m = float(np.random.uniform(200, 1400))
                dist_ref_m = float(np.random.uniform(2000, 8000))
            dist_ind_m = float(np.random.uniform(1800, 6000))
            dist_for_m = float(np.random.uniform(7000, 25000))
            is_night_val = True
            daynight = "N"
            firms_type = 3 if random.random() < 0.4 else -1
            persistence = float(np.random.uniform(0.35, 0.85))
            frp = float(np.random.uniform(18.0, 50.0))
            bright_ti4 = float(np.random.uniform(320.0, 345.0))

        elif arch == 2:
            # ⛏️ Mining Activity: close to mine (<1800m), far from heavy factories, moderate FRP
            dist_min_m = float(np.random.uniform(250, 1600))
            dist_ind_m = float(np.random.uniform(3500, 18000))
            dist_ref_m = float(np.random.uniform(12000, 40000))
            dist_agr_m = float(np.random.uniform(4000, 15000))
            dist_for_m = float(np.random.uniform(3000, 12000))
            confidence = "high"
            frp = float(np.random.uniform(16.0, 42.0))
            bright_ti4 = float(np.random.uniform(318.0, 338.0))

        elif arch == 3:
            # 🌾 Agricultural Burn: close to cropland (<800m), vegetation type 0, isolated from industry
            dist_agr_m = float(np.random.uniform(120, 750))
            dist_ind_m = float(np.random.uniform(3500, 18000))
            dist_ref_m = float(np.random.uniform(15000, 45000))
            dist_oil_m = float(np.random.uniform(15000, 45000))
            dist_min_m = float(np.random.uniform(12000, 40000))
            dist_for_m = float(np.random.uniform(5000, 22000))
            firms_type = 0
            frp = float(np.random.uniform(8.0, 32.0))
            bright_ti4 = float(np.random.uniform(312.0, 334.0))

        elif arch == 4:
            # 🌲 Forest / Natural Fire: close to forest (<1200m), high FRP, deep isolation from industry
            dist_for_m = float(np.random.uniform(150, 1100))
            dist_ind_m = float(np.random.uniform(6000, 35000))
            dist_ref_m = float(np.random.uniform(18000, 50000))
            dist_oil_m = float(np.random.uniform(18000, 50000))
            dist_agr_m = float(np.random.uniform(3500, 20000))
            dist_min_m = float(np.random.uniform(15000, 45000))
            firms_type = 0
            frp = float(np.random.uniform(32.0, 88.0))
            bright_ti4 = float(np.random.uniform(332.0, 362.0))

        elif arch == 5:
            # ♨️ Industrial Process Heat: static type 3 or low steady nighttime heat near factory
            dist_ind_m = float(np.random.uniform(200, 950))
            dist_ref_m = float(np.random.uniform(4000, 15000))
            dist_agr_m = float(np.random.uniform(4000, 15000))
            dist_for_m = float(np.random.uniform(8000, 25000))
            if random.random() < 0.6:
                firms_type = 3
                frp = float(np.random.uniform(6.0, 18.0))
            else:
                is_night_val = True
                daynight = "N"
                frp = float(np.random.uniform(5.0, 13.0))
            bright_ti4 = float(np.random.uniform(312.0, 326.0))

        else:
            # ❓ Unclassified / Noise: far from everything, low FRP/confidence
            dist_ind_m = MISSING_M
            dist_ref_m = MISSING_M
            dist_oil_m = MISSING_M
            dist_min_m = MISSING_M
            dist_agr_m = MISSING_M
            dist_for_m = MISSING_M
            dist_pow_m = MISSING_M
            confidence = "low"
            frp = float(np.random.uniform(2.0, 8.0))
            bright_ti4 = float(np.random.uniform(300.0, 310.0))

        # Build complete feature record with canonical meter and kilometer units
        out = {
            "latitude": hot_lat,
            "longitude": hot_lon,
            "frp": round(frp, 2),
            "bright_ti4": round(bright_ti4, 2),
            "bright_ti5": round(bright_ti5, 2),
            "daynight": daynight,
            "satellite": str(row.get("satellite", "VIIRS")),
            "confidence": confidence,
            "confidence_val": 1.0 if confidence == "high" else (0.0 if confidence == "low" else 0.5),
            "acq_date": str(row.get("acq_date", "2026-09-01")),
            "acq_time": str(row.get("acq_time", "0000")),
            "firms_type": firms_type,
            "is_night": is_night_val,
            "persistence_ratio": round(persistence, 2),

            # Canonical distances in METERS (used by labeling_functions)
            "distance_to_industry_m": round(dist_ind_m, 1),
            "distance_to_refinery_m": round(dist_ref_m, 1),
            "distance_to_oil_gas_m": round(dist_oil_m, 1),
            "distance_to_mining_m": round(dist_min_m, 1),
            "distance_to_agriculture_m": round(dist_agr_m, 1),
            "distance_to_forest_m": round(dist_for_m, 1),
            "distance_to_power_plant_m": round(dist_pow_m, 1),

            # Distances in KILOMETERS (used by XGBoost feature matrix)
            "dist_factory": round(dist_ind_m / 1000.0, 3),
            "dist_industrial_zone": round(dist_ind_m / 1000.0, 3),
            "dist_refinery": round(dist_ref_m / 1000.0, 3),
            "dist_oil_gas": round(dist_oil_m / 1000.0, 3),
            "dist_mining": round(dist_min_m / 1000.0, 3),
            "dist_agriculture": round(dist_agr_m / 1000.0, 3),
            "dist_forest": round(dist_for_m / 1000.0, 3),
            "dist_powerplant": round(dist_pow_m / 1000.0, 3),

            # Spatial Density Flags
            "has_refinery_5km": 1 if dist_ref_m <= 5000.0 else 0,
            "has_powerplant_5km": 1 if dist_pow_m <= 5000.0 else 0,
            "has_factory_5km": 1 if dist_ind_m <= 5000.0 else 0,
            "has_industrial_2km": 1 if dist_ind_m <= 2000.0 else 0,
            "has_forest_5km": 1 if dist_for_m <= 5000.0 else 0,
            "has_agriculture_5km": 1 if dist_agr_m <= 5000.0 else 0,
            "count_ind_5km": 3 if dist_ind_m <= 2000.0 else (1 if dist_ind_m <= 5000.0 else 0),
            "count_ref_5km": 2 if dist_ref_m <= 3000.0 else (1 if dist_ref_m <= 5000.0 else 0),
        }
        rows.append(out)

    out_df = pd.DataFrame(rows)
    # Marker column so guard check (scripts/check_data_integrity.py) can
    # detect and reject this synthetic CSV if it is accidentally passed to training.
    out_df["_is_synthetic_demo"] = True
    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
    out_df.to_csv(OUTPUT_CSV, index=False)
    print(f"\n[build_demo_dataset] Wrote {len(out_df)} SYNTHETIC demo rows -> {OUTPUT_CSV}")
    print("⚠️  This file is for DEMO/UI only. Never use for training.")
    return out_df


if __name__ == "__main__":
    build_balanced_dataset()
