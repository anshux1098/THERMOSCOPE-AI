"""
check_states_v2.py - count hotspots per state in v2 dataset
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent / "backend"))

import pandas as pd
from app.core.paths import CLASSIFIED_DATASET_PATH

states_bbox = {
    'jammu_kashmir':   (32.30, 35.50, 73.75, 80.30),
    'himachal_pradesh':(30.40, 33.20, 75.55, 79.00),
    'punjab':          (29.55, 32.55, 73.85, 76.95),
    'uttarakhand':     (28.70, 31.45, 77.55, 81.05),
    'haryana':         (27.65, 30.95, 74.45, 77.60),
    'delhi':           (28.40, 28.90, 76.85, 77.35),
    'uttar_pradesh':   (23.85, 30.45, 77.10, 84.70),
    'rajasthan':       (23.10, 30.25, 69.50, 78.25),
    'gujarat':         (20.10, 24.70, 68.15, 73.95),
    'madhya_pradesh':  (21.10, 26.90, 74.05, 82.80),
    'maharashtra':     (15.60, 22.05, 72.60, 80.90),
    'goa':             (14.90, 15.80, 73.65, 74.35),
    'karnataka':       (11.55, 18.45, 74.05, 78.60),
    'andhra_pradesh':  (12.65, 19.15, 76.75, 84.75),
    'tamil_nadu':      (8.05, 13.60, 76.20, 80.40),
    'kerala':          (8.20, 12.80, 74.85, 77.40),
    'telangana':       (15.85, 19.95, 77.25, 81.85),
    'odisha':          (17.50, 22.55, 81.50, 87.50),
    'jharkhand':       (21.95, 25.35, 83.30, 87.95),
    'chhattisgarh':    (17.85, 24.10, 80.25, 84.40),
    'west_bengal':     (21.50, 27.30, 85.85, 89.95),
    'assam':           (24.10, 27.95, 89.70, 96.05),
    'bihar':           (24.30, 27.85, 83.30, 88.30),
}

df = pd.read_csv(CLASSIFIED_DATASET_PATH)
print(f'Total hotspots in v2: {len(df)}')
print()

counts = {}
unassigned = 0
for _, r in df.iterrows():
    lat, lon = r['latitude'], r['longitude']
    found = False
    for state, (lat_min, lat_max, lon_min, lon_max) in states_bbox.items():
        if lat_min <= lat <= lat_max and lon_min <= lon <= lon_max:
            counts[state] = counts.get(state, 0) + 1
            found = True
            break
    if not found:
        unassigned += 1

print('Hotspots per state:')
for state, cnt in sorted(counts.items(), key=lambda x: -x[1]):
    print(f'  {state:20s}: {cnt:3d} hotspots')
print(f'  {"unassigned":20s}: {unassigned:3d} hotspots')
print(f'\nTotal states with hotspots: {len(counts)}')
