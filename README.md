# THERMOSCOPE-AI

**AI-Based Detection and Classification of Industrial Fires and Persistent Thermal Sources**

- **Problem:** SIH 2026 — PS ID SIH26162, NTRO, Disaster Management
- **Deadline:** 20 September 2026
- **Team:** 6 members (collaborative)
- **Stack:** Python, FastAPI, Streamlit/Leaflet, XGBoost, OpenAI

---

## What This Project Does

Satellite systems (like NASA FIRMS) detect thermal hotspots across India daily. But they can't tell whether a hotspot is a **refinery fire**, a **gas flare**, a **wildfire**, **agricultural burning**, or a **volcano** — they just show a "hot pixel."

THERMOSCOPE-AI fixes this:

1. **Fetches** FIRMS hotspot data for India (via NASA FIRMS API)
2. **Enriches** each hotspot with OpenStreetMap context (nearest industrial site, refinery, power plant, factory, etc.)
3. **Classifies** each hotspot into one of 5 categories:
   - Industrial Fire
   - Gas Flare
   - Mining
   - Agricultural Burn
   - Forest-Natural Fire
4. **Builds a persistent site registry** — clusters recurring hotspots into named monitored sites
5. **Detects anomalies** — flags sites behaving abnormally vs their historical baseline
6. **Serves** everything through a FastAPI backend + GIS frontend

---

## Current State — What's Done

### ✅ Completed

**Service Layer** (all 5 service files written and tested):

| File | What It Does | Status |
|---|---|---|
| `backend/app/services/firms_service.py` | Fetches NASA FIRMS hotspot data for India (VIIRS_SNPP_NRT, VIIRS_NOAA20_NRT). Caches to CSV (24h). Functions: `fetch_multiple_datasets()`, `get_data_availability()`, `save_firms_data()`, `load_firms_data()` | ✅ Working |
| `backend/app/services/osm_service.py` | Loads 7,204+ OSM industrial sites from GeoJSON. Query functions for 26 Indian states. `load_osm_sites()`, `query_state()`, `classify_osm_site()` | ✅ Working |
| `backend/app/services/osm_service_extra.py` | Extra Overpass queries for Haryana, Gujarat, UP. `get_state()`, `classify()`, `post()` | ✅ Working |
| `backend/app/services/historical_service.py` | Stub. Placeholder for historical hotspot data. `get_historical_hotspots()` returns `[]` | ✅ Working (stub) |
| `backend/app/services/hotspot_service.py` | **Orchestrator.** Runs the full pipeline: fetch FIRMS → load OSM context → compute features → classify (5-class rule-based) → match to site registry → detect anomalies → return structured DataFrame. `orchestrate_hotspot_classification()`, `classify_hotspot()`, `match_to_site_registry()`, `detect_anomaly()`, `compute_features()` | ✅ Working |

**Data** (already downloaded and cached):

| File | What It Contains |
|---|---|
| `data/raw/osm/osm_industrial_sites.json` | 7,204 OSM industrial sites (industrial zones, factories, power plants, refineries, oil/gas) |
| `data/raw/firms/firms_recent.csv` | 642 VIIRS hotspots (cached, 24h freshness) |

### 🔄 In Progress / Next

- **ML classifier** (XGBoost multi-class) — replace rule-based classifier when ready
- **Anomaly detection** — currently runs but returns 0 anomalies (needs historical baseline data)
- **Site registry** — basic clustering works (424 sites from 642 hotspots), needs persistent storage
- **FastAPI backend** — stub structure exists, need to wire up endpoints
- **GIS frontend** — Streamlit or React+Leaflet dashboard
- **SHAP explainability** — per-classification explanations

---

## Quick Setup (for collaborators)

### Prerequisites

- Python 3.10+ (tested with 3.11)
- Git
- NASA FIRMS API key (free — get at https://firms.modaps.eosdis.nasa.gov/api/map_key)

### Step 1: Clone

```bash
git clone https://github.com/anshux1098/THERMOSCOPE-AI.git
cd THERMOSCOPE-AI
```

### Step 2: Create Virtual Environment

```bash
python -m venv venv
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate
```

### Step 3: Install Dependencies

```bash
pip install pandas numpy scikit-learn xgboost folium requests geopy python-dotenv
pip install fastapi uvicorn pydantic sqlalchemy python-multipart
pip install shap  # for explainability (optional, for later)
```

### Step 4: Configure Environment

Copy `.env.example` to `.env` and fill in your values:

```bash
cp .env.example .env
```

Edit `.env`:
- `FIRMS_MAP_KEY` = your NASA FIRMS API key
- `DB_URL` = `sqlite:///./thermoscope.db` (or PostgreSQL+PostGIS for production)
- `CORS_ORIGINS` = `http://localhost:5173,http://localhost:8501`
- `INDIA_BBOX` = `68.0,6.0,96.0,36.0` (Indian subcontinent bounding box)
- `CACHE_HOURS` = `24`

### Step 5: Get OSM Data (one-time)

The OSM industrial sites data is already in `data/raw/osm/osm_industrial_sites.json` (7,204 sites).

To fetch fresh data or add more states:
```bash
# From the venv:
python backend/app/services/osm_service.py
```

### Step 6: Test Everything

```bash
# Test imports
python -c "from app.services.firms_service import fetch_multiple_datasets; print('OK')"
python -c "from app.services.osm_service import load_osm_sites; print('OK')"
python -c "from app.services.hotspot_service import orchestrate_hotspot_classification; print('OK')"

# Run the full pipeline (uses cached data, ~1 minute)
python backend/app/services/hotspot_service.py
```

Expected output:
```
Fetched 642 hotspots
Loaded 7204 OSM industrial sites
Orchestrator returned 642 rows
Classification distribution:
  Industrial Fire: X
  Gas Flare: Y
  ...
=== PIPELINE TEST: PASS ===
```

### Step 7: Run the FastAPI Server (when ready)

```bash
uvicorn backend.app.main:app --reload --port 8000
```

---

## Project Structure

```
THERMOSCOPE-AI/
├── backend/
│   └── app/
│       ├── __init__.py
│       ├── main.py              # FastAPI app (stub)
│       ├── core/
│       │   ├── __init__.py
│       │   ├── config.py        # Pydantic settings (FIRMS_MAP_KEY, DB_URL, etc.)
│       │   └── constants.py     # Site types, class labels, colors
│       └── services/
│           ├── __init__.py
│           ├── firms_service.py       # 🔥 NASA FIRMS API
│           ├── osm_service.py         # 🗺️ OpenStreetMap context
│           ├── osm_service_extra.py   # 🗺️ Extra OSM queries (Haryana, Gujarat, UP)
│           ├── historical_service.py  # 📅 Historical data (stub)
│           └── hotspot_service.py     # 🧠 Orchestrator (full pipeline)
├── data/
│   ├── raw/
│   │   ├── firms/              # FIRMS CSV cache
│   │   └── osm/                # OSM GeoJSON
│   └── processed/              # Pipeline outputs (CSV/GeoJSON)
├── .env                        # API keys, config (DO NOT commit)
├── .env.example                # Template for .env
├── .gitignore
└── README.md
```

---

## How Classification Works (Current — Rule-Based)

The `hotspot_service.py` `classify_hotspot()` function uses these rules (Blueprint §5 taxonomy):

1. **Industrial Fire**: distance to industrial site < 1km AND FRP > 50 MW
2. **Gas Flare**: distance to oil/gas facility < 2km AND persistent detection
3. **Mining**: distance to mining site < 2km
4. **Agricultural Burn**: land-cover = cropland AND seasonal AND low FRP
5. **Forest-Natural Fire**: default — far from industrial sites, no agricultural context

Each classification gets:
- `classification`: class label
- `confidence`: 0–1 score
- `explanation`: human-readable reason

**This is rule-based for now.** The ML classifier (XGBoost) will replace it when trained.

---

## How to Contribute / Make Changes

### Adding a New Feature

1. Pick a service file or create a new one under `backend/app/services/`
2. Write the function
3. Import it in `hotspot_service.py` if it's part of the pipeline
4. Test it: `python -c "from app.services.your_module import your_function; print('OK')"`
5. Update this README if needed

### Running Tests

```bash
# Quick import test
python -c "
from app.services.firms_service import fetch_multiple_datasets
from app.services.osm_service import load_osm_sites
from app.services.hotspot_service import orchestrate_hotspot_classification
print('All imports OK')
"

# Pipeline test (uses cached data)
python backend/app/services/hotspot_service.py
```

### Adding More OSM States

Edit `backend/app/services/osm_service.py` — add your state's bounding box to the `INDIA_STATES` dictionary, then run:
```bash
python -c "from app.services.osm_service import query_all_states; query_all_states()"
```

### Changing Classification Rules

Edit `backend/app/services/hotspot_service.py` → `classify_hotspot()` function. The rules are clearly commented there.

### Adding ML Classifier

1. Create `backend/app/services/ml_classifier.py`
2. Implement `train_model()`, `predict()`, `explain()` using XGBoost
3. Import in `hotspot_service.py` and use instead of rule-based classifier
4. Save trained model to `models/` directory

---

## Data Flow (How It All Fits Together)

```
FIRMS API (NASA)          OSM Overpass API
       │                        │
       ▼                        ▼
firms_service.py          osm_service.py
       │                        │
       ▼                        ▼
       └───── hotspot_service.py (ORCHESTRATOR) ─────┘
                       │
                       ├── compute_features()    → distance, land-use, persistence
                       ├── classify_hotspot()    → 5-class rule-based
                       ├── match_to_site_registry() → cluster into sites
                       ├── detect_anomaly()      → baseline deviation check
                       └── return structured DataFrame
```

---

## Key Files to Know

| File | Why It Matters |
|---|---|
| `backend/app/services/hotspot_service.py` | **The main orchestrator.** If you want to understand or modify the pipeline, start here. |
| `backend/app/services/firms_service.py` | FIRMS data fetching. Change cache duration, datasets, or API here. |
| `backend/app/services/osm_service.py` | OSM site loading and querying. Add states, modify site types here. |
| `backend/app/core/config.py` | All configuration (API keys, DB URL, bounding box) lives here. |
| `.env` | Your personal config (API keys). NEVER commit this file. |

---

## Known Limitations

- **Rule-based classifier only** — ML classifier not yet trained
- **Anomaly detection returns 0 anomalies** — needs historical baseline data per site
- **No persistent database** — in-memory only for now (SQLite/PostGIS planned)
- **No frontend** — backend only, GIS dashboard pending
- **OSM data incomplete** — India industrial sites poorly mapped in some regions (documented limitation)

---

## Resources

- **NASA FIRMS API:** https://firms.modaps.eosdis.nasa.gov/api/map_key
- **OpenStreetMap Overpass API:** https://overpass-api.de/
- **SIH 2026 Problem Statement (SIH26162):** See `docs/` or NTRO SIH portal
- **Blueprint PDF:** `SIH26162_Blueprint.pdf` in repo root

---

## License

This project is part of SIH 2026 (Smart India Hackathon). All code is open-source for collaboration.

---

## Contact

- **Repo:** https://github.com/anshux1098/THERMOSCOPE-AI
- **Issues:** Use GitHub Issues for bugs, questions, feature requests
