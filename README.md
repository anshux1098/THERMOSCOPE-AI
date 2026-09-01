# THERMOSCOPE-AI — SIH26162

**AI-Based Detection & Classification of Industrial Fires and Persistent Thermal Sources (NTRO)**

AGNI-DRISHTI — Garam pixels se actionable intelligence tak.

## 5-Class Taxonomy
`Industrial Fire` | `Gas Flare` | `Mining` | `Agricultural Burn` | `Forest-Natural Fire`

Pipeline: `FIRMS` → `OSM + land-cover context` → `Feature engineering` → `Classification` → `Site registry match` → `Anomaly baseline check` → `Structured output`

## Quick Start

```powershell
# 1. Clone
git clone https://github.com/<your-username>/THERMOSCOPE-AI.git
cd THERMOSCOPE-AI

# 2. Env (Windows)
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt

# 3. Secrets
Copy-Item .env.example .env
# edit .env and set FIRMS_MAP_KEY from https://firms.modaps.eosdis.nasa.gov/api/area/

# 4. Run orchestrator
python -m app.services.hotspot_service
# also: python backend/app/services/hotspot_service.py

# 5. Individual services
python -c "from app.services.firms_service import fetch_multiple_datasets; print(fetch_multiple_datasets(days=3).head())"
python -c "from app.services.osm_service import load_osm_sites; print(len(load_osm_sites()))"
```

## Project Structure
```
backend/app/
  core/config.py, constants.py
  services/firms_service.py      # FIRMS fetch (fetch_multiple_datasets)
  services/osm_service.py        # OSM Overpass (load_osm_sites)
  services/osm_service_extra.py  # 3-state helper
  services/historical_service.py # stub
  services/hotspot_service.py    # ORCHESTRATOR (classify, registry, anomaly)
data/
  raw/firms/firms_recent.csv
  raw/osm/osm_industrial_sites.json (20k sites)
  processed/hotspots/
```

## Collaboration
```powershell
git checkout -b feat/<your-feature>
# make changes
git add .
git commit -m "feat: <desc>"
git push -u origin feat/<your-feature>
# → open Pull Request on GitHub
```

Ask maintainer to add you as **Collaborator** (Settings → Collaborators) if repo is Private.

## Blueprint
See `C:\Users\ASUS\Downloads\SIH26162_Blueprint.pdf` (local) — §10 Architecture.

## Tech
Python 3.11, pandas, requests, scikit-learn, FastAPI (future), PostGIS (future)
