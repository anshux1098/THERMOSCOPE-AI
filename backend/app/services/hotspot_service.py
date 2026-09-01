"""
hotspot_service.py
ORCHESTRATOR for SIH26162 — AI-Based Detection and Classification of Industrial Fires
and Persistent Thermal Sources (NTRO).

Pipeline: FIRMS → OSM + land-cover context → Feature engineering → Classification →
Site registry match → Anomaly baseline check → Store → Return structured output

5-class taxonomy (Blueprint): Industrial Fire, Gas Flare, Mining, Agricultural Burn, Forest-Natural Fire

This file keeps feature engineering, classification, registry, anomaly logic INSIDE
hotspot_service.py (future separation into dedicated modules per blueprint §10).
"""
import math
import sys
import uuid
from pathlib import Path
# Ensure `app` package is found when run as `python backend/app/services/hotspot_service.py`
if str(Path(__file__).resolve().parents[3]) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from typing import Optional, Tuple, Dict, List

import pandas as pd
import numpy as np

# Building blocks
try:
    from app.services import firms_service as _firms_service  # noqa: F401
    from app.services import osm_service as _osm_service  # noqa: F401
    from app.services import historical_service as _historical_service  # noqa: F401
except Exception:
    _firms_service = None
    _osm_service = None
    _historical_service = None

# haversine — try app.geo.distance, else inline
try:
    from app.geo.distance import haversine_km  # type: ignore
except Exception:
    def haversine_km(lat1, lon1, lat2, lon2):
        R = 6371.0
        dlat = math.radians(lat2 - lat1)
        dlon = math.radians(lon2 - lon1)
        a = math.sin(dlat / 2) ** 2 + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon / 2) ** 2
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
        return R * c

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _min_distance_to_type(lat, lon, sites, allowed_types):
    """Minimum haversine distance (km) from (lat,lon) to any site whose site_type in allowed_types."""
    if not sites:
        return 999.0
    min_d = 999.0
    for s in sites:
        if s.get("site_type") in allowed_types:
            try:
                d = haversine_km(lat, lon, float(s["lat"]), float(s["lon"]))
                if d < min_d:
                    min_d = d
            except Exception:
                continue
    return min_d

def _land_use_context(lat, lon, sites) -> str:
    """
    Infer land_use_context from nearest OSM site type.
    Blueprint land-cover: forest / cropland / built-up / industrial.
    Fallback: use site_type → mapping; else 'unknown'.
    """
    if not sites:
        return "unknown"
    # find nearest site overall
    best_type = "unknown"
    best_d = 999.0
    for s in sites:
        try:
            d = haversine_km(lat, lon, float(s["lat"]), float(s["lon"]))
            if d < best_d:
                best_d = d
                best_type = s.get("site_type", "unknown")
        except Exception:
            continue
    mapping = {
        "factory": "industrial",
        "refinery": "industrial",
        "industrial_zone": "industrial",
        "power_plant": "industrial",
        "oil_gas": "industrial",
        "power_infrastructure": "industrial",
        "other_industrial": "industrial",
        "mining": "mining",
        "volcano": "forest",  # natural
    }
    # if far from any site (>5km) consider natural/forest vs cropland heuristic
    if best_d > 10:
        return "forest"  # default natural
    return mapping.get(best_type, "unknown")

# ---------------------------------------------------------------------------
# 5. compute_features
# ---------------------------------------------------------------------------

def compute_features(hotspot_row, osm_sites) -> Dict:
    """
    Compute features for a single hotspot:
      distance_to_industrial_km, distance_to_refinery_km, distance_to_powerplant_km,
      distance_to_factory_km, distance_to_volcano_km, land_use_context,
      persistence_count, historical_frp_baseline
    """
    # lat/lon may be 'latitude'/'longitude' or 'lat'/'lon'
    lat = hotspot_row.get("latitude", hotspot_row.get("lat", None))
    lon = hotspot_row.get("longitude", hotspot_row.get("lon", None))
    if lat is None or lon is None:
        return {
            "distance_to_industrial_km": 999.0,
            "distance_to_refinery_km": 999.0,
            "distance_to_powerplant_km": 999.0,
            "distance_to_factory_km": 999.0,
            "distance_to_volcano_km": 999.0,
            "distance_to_oil_gas_km": 999.0,
            "distance_to_mining_km": 999.0,
            "land_use_context": "unknown",
            "persistence_count": int(hotspot_row.get("persistence_count", 1) or 1),
            "historical_frp_baseline": None,
        }
    lat = float(lat); lon = float(lon)
    sites = osm_sites or []
    industrial_types = {"factory", "refinery", "industrial_zone", "power_plant", "oil_gas", "mining", "other_industrial", "power_infrastructure"}
    dist_industrial = _min_distance_to_type(lat, lon, sites, industrial_types)
    dist_refinery = _min_distance_to_type(lat, lon, sites, {"refinery"})
    dist_powerplant = _min_distance_to_type(lat, lon, sites, {"power_plant", "power_infrastructure"})
    dist_factory = _min_distance_to_type(lat, lon, sites, {"factory"})
    dist_volcano = _min_distance_to_type(lat, lon, sites, {"volcano"})
    dist_oil_gas = _min_distance_to_type(lat, lon, sites, {"oil_gas", "refinery"})
    dist_mining = _min_distance_to_type(lat, lon, sites, {"mining"})
    land_use = _land_use_context(lat, lon, sites)
    # allow palette: if hotspot already has land-cover column, prefer it
    if "land_use_context" in hotspot_row and pd.notna(hotspot_row.get("land_use_context")):
        land_use = str(hotspot_row.get("land_use_context"))
    elif "land_cover" in hotspot_row and pd.notna(hotspot_row.get("land_cover")):
        land_use = str(hotspot_row.get("land_cover"))

    # persistence_count could be supplied externally; default 1
    persistence_count = hotspot_row.get("persistence_count", hotspot_row.get("persistence_ratio", None))
    if persistence_count is None:
        persistence_count = 1
    else:
        try:
            # if persistence_ratio 0-1, convert to count proxy?
            pc = float(persistence_count)
            if 0 <= pc <= 1:
                # ratio → keep ratio as is for classifier, but also expose count
                persistence_count = pc
            else:
                persistence_count = int(pc)
        except Exception:
            persistence_count = 1

    historical_frp_baseline = hotspot_row.get("historical_frp_baseline", None)
    # try historical_data dict if passed via hotspot_row
    if historical_frp_baseline is None and "baseline_mean" in hotspot_row:
        historical_frp_baseline = {"mean": hotspot_row.get("baseline_mean"), "std": hotspot_row.get("baseline_std")}

    return {
        "distance_to_industrial_km": float(dist_industrial),
        "distance_to_refinery_km": float(dist_refinery),
        "distance_to_powerplant_km": float(dist_powerplant),
        "distance_to_factory_km": float(dist_factory),
        "distance_to_volcano_km": float(dist_volcano),
        "distance_to_oil_gas_km": float(dist_oil_gas),
        "distance_to_mining_km": float(dist_mining),
        "land_use_context": land_use,
        "persistence_count": persistence_count,
        "historical_frp_baseline": historical_frp_baseline,
    }

# ---------------------------------------------------------------------------
# 2. classify_hotspot
# ---------------------------------------------------------------------------

def classify_hotspot(row, osm_context, historical_baseline=None) -> Tuple[str, float, str]:
    """
    Rule-based classification for a single hotspot (Blueprint 5-class taxonomy).
    Returns (classification, confidence, explanation)
    """
    feats = osm_context if isinstance(osm_context, dict) else {}
    # extract needed values
    dist_industrial = float(feats.get("distance_to_industrial_km", 999))
    dist_oil_gas = float(feats.get("distance_to_oil_gas_km", feats.get("distance_to_refinery_km", 999)))
    dist_mining = float(feats.get("distance_to_mining_km", 999))
    land_use = str(feats.get("land_use_context", "unknown")).lower()
    persistence = feats.get("persistence_count", 1)

    # persistence flag: >0.4 ratio or count>1 or historical multiple days
    if isinstance(persistence, float) and 0 <= persistence <= 1:
        is_persistent = persistence > 0.4
    else:
        try:
            is_persistent = int(persistence) > 1
        except Exception:
            is_persistent = False
    # also consider historical_data presence
    if isinstance(historical_baseline, dict) and historical_baseline.get("count", 0) > 3:
        is_persistent = True

    frp = float(row.get("frp", row.get("FRP", 0)) or 0)
    # FRP thresholds
    high_frp = frp > 50

    # Rule 1: Mining — distance to mining <2km
    if dist_mining < 2.0:
        return ("Mining", 0.85, f"Mining site within {dist_mining:.2f} km, FRP={frp:.1f} MW")

    # Rule 2: Industrial Fire — distance <1km AND high FRP
    if dist_industrial < 1.0 and high_frp:
        return ("Industrial Fire", 0.90, f"Industrial site {dist_industrial:.2f} km away, high FRP {frp:.1f} MW (>50)")

    # Rule 3: Gas Flare — oil/gas <2km AND persistent
    if dist_oil_gas < 2.0 and is_persistent:
        return ("Gas Flare", 0.88, f"Oil/gas facility {dist_oil_gas:.2f} km away, persistent (count={persistence})")

    # Rule 4: Agricultural Burn — cropland AND low FRP (and not persistent)
    if land_use in ("cropland", "agricultural", "crop") and frp < 30 and not is_persistent:
        return ("Agricultural Burn", 0.70, f"Cropland context, low FRP {frp:.1f} MW, ephemeral")

    # Rule 5: Forest-Natural Fire — forest AND low persistence AND no industrial context
    if land_use in ("forest", "natural") and not is_persistent and dist_industrial > 2.0:
        return ("Forest-Natural Fire", 0.75, f"Forest land-cover, ephemeral, distant from industry ({dist_industrial:.1f} km)")

    # Additional forest fallback with low FRP
    if land_use == "forest" and frp < 30:
        return ("Forest-Natural Fire", 0.65, f"Forest context, low FRP {frp:.1f} MW")

    # Default
    return ("Forest-Natural Fire", 0.50, f"Default: no strong industrial/mining/flare signature (dist_ind={dist_industrial:.1f} km, FRP={frp:.1f})")


# ---------------------------------------------------------------------------
# 4. detect_anomaly
# ---------------------------------------------------------------------------

def detect_anomaly(hotspot_row, historical_baseline) -> Tuple[bool, float, float, float, float]:
    """
    Check if current FRP deviates significantly from historical baseline.
    anomaly_flag = True if current_frp > baseline_mean + 3*baseline_std
    new_site_flag = True if no historical baseline exists (handled via baseline_mean/std=0)
    Returns (anomaly_flag, baseline_mean, baseline_std, current_frp, deviation)
    """
    current_frp = float(hotspot_row.get("frp", hotspot_row.get("FRP", 0)) or 0)
    if historical_baseline is None or (isinstance(historical_baseline, dict) and historical_baseline.get("mean") is None):
        # no history → new site, not anomaly by deviation but flagged as new
        return (False, 0.0, 0.0, current_frp, 0.0)

    if isinstance(historical_baseline, dict):
        baseline_mean = float(historical_baseline.get("mean", 0) or 0)
        baseline_std = float(historical_baseline.get("std", 0) or 0)
        # also handle historical_baseline as tuple/list
    elif isinstance(historical_baseline, (list, tuple)) and len(historical_baseline) >= 2:
        baseline_mean = float(historical_baseline[0] or 0)
        baseline_std = float(historical_baseline[1] or 0)
    else:
        baseline_mean = 0.0
        baseline_std = 0.0

    if baseline_std == 0:
        # if no variance, use mean comparison
        if baseline_mean == 0:
            return (False, baseline_mean, baseline_std, current_frp, 0.0)
        deviation = (current_frp - baseline_mean) / max(1e-6, baseline_mean)
        anomaly = current_frp > baseline_mean * 1.5  # 50% above mean if no std
        return (bool(anomaly), baseline_mean, baseline_std, current_frp, float(deviation))

    deviation = (current_frp - baseline_mean) / max(1e-6, baseline_std)
    anomaly_flag = current_frp > (baseline_mean + 3 * baseline_std)
    return (bool(anomaly_flag), baseline_mean, baseline_std, current_frp, float(deviation))

# ---------------------------------------------------------------------------
# 3. match_to_site_registry
# ---------------------------------------------------------------------------

def match_to_site_registry(hotspots_df, existing_sites=None) -> pd.DataFrame:
    """
    Cluster nearby hotspots (2km threshold) into persistent sites.
    Match new hotspots to existing sites by proximity.
    Create new site records if no match.
    Returns DataFrame with site_id, site_type, detection_count, first_detected, last_detected
    plus mapping for hotspots (hotspot_id → site_id).
    """
    if hotspots_df is None or hotspots_df.empty:
        return pd.DataFrame(columns=["site_id", "site_type", "detection_count", "first_detected", "last_detected", "centroid_lat", "centroid_lon"])

    # normalize lat/lon cols
    lat_col = "latitude" if "latitude" in hotspots_df.columns else ("lat" if "lat" in hotspots_df.columns else None)
    lon_col = "longitude" if "longitude" in hotspots_df.columns else ("lon" if "lon" in hotspots_df.columns else None)
    if lat_col is None or lon_col is None:
        raise ValueError("hotspots_df must contain latitude/longitude columns")

    # prepare existing sites list for matching
    existing = []
    if existing_sites is not None:
        if isinstance(existing_sites, pd.DataFrame) and not existing_sites.empty:
            for _, r in existing_sites.iterrows():
                existing.append({"site_id": r.get("site_id"), "lat": float(r.get("centroid_lat", r.get("latitude", 0))), "lon": float(r.get("centroid_lon", r.get("longitude", 0))), "site_type": r.get("site_type", "unknown")})
        elif isinstance(existing_sites, list):
            existing = existing_sites

    # simple greedy clustering 2km threshold
    clusters = []  # list of dict {site_id, centroid_lat, centroid_lon, members: [indices], site_type}
    next_site_counter = 1
    # if existing sites exist, seed clusters with them
    for es in existing:
        clusters.append({
            "site_id": es.get("site_id", f"SITE-{next_site_counter:04d}"),
            "centroid_lat": float(es["lat"]),
            "centroid_lon": float(es["lon"]),
            "members": [],
            "site_type": es.get("site_type", "unknown"),
            "first_detected": None,
            "last_detected": None,
        })
        next_site_counter += 1

    # assign each hotspot to nearest cluster within 2km else new
    hotspot_site_ids = []
    for idx, row in hotspots_df.iterrows():
        lat = float(row[lat_col]); lon = float(row[lon_col])
        best = None; best_d = 999.0; best_idx = -1
        for ci, c in enumerate(clusters):
            d = haversine_km(lat, lon, c["centroid_lat"], c["centroid_lon"])
            if d < best_d:
                best_d = d; best = c; best_idx = ci
        if best is not None and best_d <= 2.0:
            # match existing cluster, update centroid incremental
            n = len(best["members"])
            best["centroid_lat"] = (best["centroid_lat"] * n + lat) / (n + 1)
            best["centroid_lon"] = (best["centroid_lon"] * n + lon) / (n + 1)
            best["members"].append(idx)
            hotspot_site_ids.append(best["site_id"])
        else:
            # new site
            site_id = f"SITE-{next_site_counter:04d}"
            next_site_counter += 1
            new_c = {
                "site_id": site_id,
                "centroid_lat": lat,
                "centroid_lon": lon,
                "members": [idx],
                "site_type": row.get("site_type", row.get("nearest_site_type", "unknown")) if "site_type" in hotspots_df.columns or "nearest_site_type" in hotspots_df.columns else "unknown",
                "first_detected": None,
                "last_detected": None,
            }
            clusters.append(new_c)
            hotspot_site_ids.append(site_id)

    # build site registry rows
    site_rows = []
    for c in clusters:
        if not c["members"]:
            continue  # existing site with no new members — still keep? include if needed
            # skip empty existing without new detections for now
        # gather member rows for dates
        members = hotspots_df.loc[c["members"]] if c["members"] else pd.DataFrame()
        # site_type: most common classification if available else stored
        if not members.empty and "classification" in members.columns:
            site_type = members["classification"].mode().iloc[0] if not members["classification"].mode().empty else c["site_type"]
        elif not members.empty and "site_type" in members.columns:
            site_type = members["site_type"].mode().iloc[0] if not members["site_type"].mode().empty else c["site_type"]
        else:
            site_type = c["site_type"]
        detection_count = len(c["members"])
        # dates
        first_detected = None; last_detected = None
        date_col = None
        for cand in ["acq_date", "acq_time", "date", "first_detected"]:
            if cand in hotspots_df.columns:
                date_col = cand
                break
        if date_col and not members.empty:
            try:
                dates = pd.to_datetime(members[date_col], errors="coerce")
                if dates.notna().any():
                    first_detected = dates.min()
                    last_detected = dates.max()
            except Exception:
                pass
        site_rows.append({
            "site_id": c["site_id"],
            "site_type": site_type,
            "detection_count": detection_count,
            "first_detected": first_detected,
            "last_detected": last_detected,
            "centroid_lat": c["centroid_lat"],
            "centroid_lon": c["centroid_lon"],
        })

    return pd.DataFrame(site_rows)

# ---------------------------------------------------------------------------
# 1. orchestrate_hotspot_classification
# ---------------------------------------------------------------------------

def orchestrate_hotspot_classification(hotspots_df, osm_sites=None, historical_data=None) -> pd.DataFrame:
    """
    Orchestrator: FIRMS → OSM context → Feature engineering → Classification → Registry match → Anomaly check.
    Returns structured DataFrame with columns:
      hotspot_id, latitude, longitude, classification, confidence, explanation,
      site_id, site_type, anomaly_flag, baseline_mean, baseline_std, current_frp, deviation
    """
    if hotspots_df is None or hotspots_df.empty:
        cols = ["hotspot_id","latitude","longitude","classification","confidence","explanation","site_id","site_type","anomaly_flag","baseline_mean","baseline_std","current_frp","deviation"]
        return pd.DataFrame(columns=cols)

    # ensure copy
    df = hotspots_df.copy().reset_index(drop=True)

    # normalize lat/lon
    if "latitude" not in df.columns and "lat" in df.columns:
        df["latitude"] = df["lat"]
    if "longitude" not in df.columns and "lon" in df.columns:
        df["longitude"] = df["lon"]

    # load OSM sites if not provided
    if osm_sites is None:
        try:
            if _osm_service and hasattr(_osm_service, "load_osm_sites"):
                osm_sites = _osm_service.load_osm_sites()
        except Exception:
            osm_sites = None
        if osm_sites is None:
            osm_sites = []
    if osm_sites is None:
        osm_sites = []

    # historical_data: dict site_id or (lat,lon) → baseline {mean,std,count} or DataFrame
    # Normalize to dict keyed by (rounded lat,lon) or site
    hist_map: Dict = {}
    if isinstance(historical_data, dict):
        hist_map = historical_data
    elif isinstance(historical_data, pd.DataFrame) and not historical_data.empty:
        # expect columns site_id, baseline_mean, baseline_std
        for _, r in historical_data.iterrows():
            sid = r.get("site_id")
            if sid:
                hist_map[str(sid)] = {"mean": r.get("baseline_mean", r.get("mean", 0)), "std": r.get("baseline_std", r.get("std", 0)), "count": r.get("count", 1)}
    # also try historical_service
    if not hist_map and _historical_service:
        try:
            # stub returns []
            pass
        except Exception:
            pass

    # step b+c : compute features + classify per hotspot, also anomaly
    classifications = []
    confidences = []
    explanations = []
    anomaly_flags = []
    baseline_means = []
    baseline_stds = []
    current_frps = []
    deviations = []
    feature_cache = []

    for idx, row in df.iterrows():
        feats = compute_features(row, osm_sites)
        feature_cache.append(feats)
        # historical baseline lookup: try site proximity or direct row's baseline
        # for now, per-hotspot baseline from hist_map by nearest lat/lon key (simple)
        baseline = None
        # check row has baseline
        if "historical_frp_baseline" in row and isinstance(row["historical_frp_baseline"], dict):
            baseline = row["historical_frp_baseline"]
        # else try hist_map with hotspot coordinate rounded
        if baseline is None:
            key = f"{round(float(row['latitude']),2)},{round(float(row['longitude']),2)}"
            baseline = hist_map.get(key)
        if baseline is None and hist_map:
            # try site-based: use first entry if small map (demo)
            # fallback to global mean
            if len(hist_map) == 1:
                baseline = list(hist_map.values())[0]
        cls, conf, expl = classify_hotspot(row, feats, baseline)
        classifications.append(cls)
        confidences.append(conf)
        explanations.append(expl)

        anomaly_flag, bm, bs, cf, dev = detect_anomaly(row, baseline)
        # also set new_site_flag as anomaly elaboration: if baseline is None, not anomaly but new
        anomaly_flags.append(bool(anomaly_flag))
        baseline_means.append(float(bm))
        baseline_stds.append(float(bs))
        current_frps.append(float(cf))
        deviations.append(float(dev))

    # attach to df
    df["classification"] = classifications
    df["confidence"] = confidences
    df["explanation"] = explanations
    df["anomaly_flag"] = anomaly_flags
    df["baseline_mean"] = baseline_means
    df["baseline_std"] = baseline_stds
    df["current_frp"] = current_frps
    df["deviation"] = deviations

    # step d: site registry match (2km threshold)
    # Need to pass classification for site_type
    site_registry = match_to_site_registry(df, existing_sites=None)
    # map hotspot → site_id via nearest cluster (recompute mapping by proximity to registry centroids)
    # Simple: for each hotspot, find nearest registry centroid within 2km
    hotspot_site_ids = []
    hotspot_site_types = []
    for _, row in df.iterrows():
        lat = float(row["latitude"]); lon = float(row["longitude"])
        best_id = None; best_type = "unknown"; best_d = 999.0
        for _, srow in site_registry.iterrows():
            d = haversine_km(lat, lon, float(srow["centroid_lat"]), float(srow["centroid_lon"]))
            if d < best_d:
                best_d = d; best_id = srow["site_id"]; best_type = srow["site_type"]
        if best_id is not None and best_d <= 2.0:
            hotspot_site_ids.append(best_id)
            hotspot_site_types.append(best_type)
        else:
            # fallback: generate per-hotspot site
            hotspot_site_ids.append(f"SITE-{uuid.uuid4().hex[:4]}")
            hotspot_site_types.append(row.get("classification", "unknown"))
    df["site_id"] = hotspot_site_ids
    df["site_type"] = hotspot_site_types

    # hotspot_id
    if "hotspot_id" not in df.columns:
        df["hotspot_id"] = [f"HS-{i:05d}" for i in range(len(df))]

    # final column order as per spec
    out_cols = ["hotspot_id","latitude","longitude","classification","confidence","explanation","site_id","site_type","anomaly_flag","baseline_mean","baseline_std","current_frp","deviation"]
    # ensure all cols exist
    for c in out_cols:
        if c not in df.columns:
            df[c] = None
    result = df[out_cols].copy()
    return result

# ---------------------------------------------------------------------------
# 6. CLI demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("="*60)
    print("SIH26162 Hotspot Orchestrator — CLI Demo")
    print("="*60)
    # Load FIRMS data — try multiple cache paths, avoid network in demo
    df = None
    try:
        from app.services.firms_service import load_firms_data  # type: ignore
        for p in ["data/raw/firms/firms_recent.csv", "data/raw/firms_recent.csv", "C:/Thermoscope-Ai/data/raw/firms/firms_recent.csv"]:
            df = load_firms_data(p)
            if df is not None and not df.empty:
                print(f"[1] Loaded {len(df)} FIRMS hotspots from {p}")
                print(f"  Columns: {list(df.columns)[:10]}")
                break
        if df is None or df.empty:
            print("  No FIRMS cache found — using synthetic demo hotspots (skip network)")
            df = None
    except Exception as e:
        print(f"  firms_service import failed: {e}")
        df = None

    # limit real FIRMS data to 20 rows for quick demo (642 rows × 20k OSM = heavy)
    if df is not None and not df.empty and len(df) > 20:
        print(f"  Sampling 20 rows from {len(df)} for demo speed")
        df = df.head(20).copy()

    if df is None or df.empty:
        # synthetic demo: 5 hotspots covering 5 classes near known sites
        df = pd.DataFrame([
            {"latitude": 22.47, "longitude": 70.05, "frp": 80.0, "confidence": "h", "daynight": "N", "acq_date": "2024-01-01", "type": 0},  # refinery → Industrial Fire
            {"latitude": 19.43, "longitude": 71.34, "frp": 45.0, "confidence": "h", "acq_date": "2024-01-02", "type": 0, "persistence_count": 5},  # oil_gas persistent → Gas Flare
            {"latitude": 24.08, "longitude": 82.66, "frp": 20.0, "acq_date": "2024-01-03", "type": 0},  # singrauli mining → Mining
            {"latitude": 30.90, "longitude": 75.85, "frp": 12.0, "acq_date": "2024-01-04", "type": 0},  # ludhiana cropland → Ag Burn
            {"latitude": 22.00, "longitude": 78.50, "frp": 15.0, "acq_date": "2024-01-05", "type": 0},  # forest → Forest-Natural
        ])
        print(f"  Synthetic DF: {len(df)} rows")

    # Load OSM sites
    osm_sites = []
    try:
        from app.services.osm_service import load_osm_sites as _load_osm  # type: ignore
        for p in ["data/raw/osm/osm_industrial_sites.json", "data/raw/osm_industrial_sites.json", "C:/Thermoscope-Ai/data/raw/osm/osm_industrial_sites.json"]:
            osm_sites = _load_osm(p)
            if osm_sites:
                print(f"[2] Loaded {len(osm_sites)} OSM sites from {p}")
                from collections import Counter
                print(f"  Types: {Counter(s.get('site_type') for s in osm_sites[:20])}")
                break
        if not osm_sites:
            print("  No OSM sites found — using fallback demo sites")
            raise FileNotFoundError
    except Exception as e:
        print(f"  OSM load failed/empty: {e}")
        osm_sites = [
            {"lat": 22.47, "lon": 70.05, "site_type": "refinery", "name": "Jamnagar Refinery"},
            {"lat": 19.43, "lon": 71.34, "site_type": "oil_gas", "name": "Bombay High"},
            {"lat": 24.08, "lon": 82.66, "site_type": "mining", "name": "Singrauli Coalfield"},
            {"lat": 30.90, "lon": 75.85, "site_type": "industrial_zone", "name": "Ludhiana IE"},
        ]
        print(f"  Using fallback {len(osm_sites)} demo OSM sites")

    # Run orchestrator
    print("\n[3] Running orchestrate_hotspot_classification...")
    try:
        result = orchestrate_hotspot_classification(df, osm_sites=osm_sites, historical_data=None)
        print(f"  Result: {len(result)} rows")
        print(result[["hotspot_id","latitude","longitude","classification","confidence","site_id","anomaly_flag","current_frp"]].head(10).to_string(index=False))
        print("\n  Class distribution:")
        print(result["classification"].value_counts().to_dict())
        if "anomaly_flag" in result.columns:
            print(f"  Anomalies flagged: {int(result['anomaly_flag'].sum())}")
    except Exception as e:
        import traceback
        print(f"  Orchestration failed: {e}")
        traceback.print_exc()

    print("\nDemo complete.")
