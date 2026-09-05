# GEOSPATIAL AUDIT REPORT — THERMOSCOPE-AI

**Date:** 2026-09-06
**Scope:** Forensic audit of the spatial intelligence pipeline to determine whether search-radius logic / spatial feature generation contributes to the extreme `unclassified` class imbalance (1208/1268 rows, 95.2%).
**Constraint honored:** No distance threshold changed, no SMOTE/oversampling, no model change, no hybrid-engine redesign. All measurement was read-only diagnostic instrumentation (`scripts/geospatial_audit.py`, `scripts/geospatial_audit_sim.py`).
**Artifacts:** `reports/audit/*.csv`, `reports/audit/*.png`.

---

## 1. EXECUTIVE SUMMARY

**VERDICT: PARTIALLY.**

The measured answer to the question "does the spatial search-radius logic cause the class imbalance?" is **PARTIALLY — and most of the imbalance is NOT caused by the radius values themselves**. The radius math, units, coordinate order, and nearest-distance calculation are **correct and verified to 0.1 m**. The discovery radius (15 km) is **rarely the binding constraint**. The dominant real causes are:

1. **HARDWARE/L2CA-DATA BUG: the forest/agriculture OSM cache is never loaded.** `scripts/build_real_dataset.py` reads only the industrial cache (`data/raw/osm/osm_industrial_sites.json`); the separate existing cache `data/raw/osm/osm_forest_agriculture.json` (3,083 forest + 708 agriculture sites) is never merged, so `distance_to_forest_m`, `dist_forest`, `has_forest_5km`, `dist_agriculture` etc. are **constant** (999 km / 0) for all 1,268 rows. Result: *zero* forest/agriculture LF votes, *zero* evidence.
2. **EVIDENCE-THRESHOLD suppression (behavior by design, verified harmful):** even if the forest/agriculture data had been merged, only **6 of 177** forest-proximate hotspots and **1 of 34** agriculture-proximate hotspots pass the LFs' thermal gates (FRP ≥ 5 MW; 65.1% of all hotspots have FRP < 5 MW, median 3.55 MW). The radius is not the limiter here — the FRP gates are.
3. **Missing FIRMS `type` column:** the VIIRS CSV has no `type` field, so `firms_type` is always `None`; every LF branch that consumes `firms_type` (vegetation detection, static/industrial source) fires at a structural discount.
4. **Genuine OSM sparsity for minority infra:** only 22 refinery sites, 15 oil/gas sites, and 94 mining sites exist in the entire India cache. At the 2 km classification threshold these capture 15 / 5 / 7 hotspots respectively (1.2% / 0.4% / 0.6%). No reasonable radius fixes this — the infrastructure rows themselves are few.
5. Ran conditionals and **misnamed/confounding features** further suppress votes and diverge live vs batch predictions (`count_ind_5km`/`count_ref_5km` are distance-bucket codes, not counts; live pipeline hardcodes them to 0).

**Decomposition of the 1,208 unclassified rows (root cause):**

| Root cause                                              | Rows | % of unclassified |
|---------------------------------------------------------|-----:|------------------:|
| All LFs blocked at the **spatial gate** (no entity found within any LF threshold) | 1,157 | **95.78 %** |
| — *of which forest/agriculture features were disabled by the cache-merge bug* | *~211* | *~17.5 %* |
| Entity within threshold but **secondary evidence failed** | 51 | 4.22 % |
| — FRP below 5 MW thermal minimum                         | 49 | 4.06 % |
| — Night evidence unavailable                             | 4  | 0.33 % |
| — **FIRMS `type` missing (firms_type)** affects the whole class | 51 | 4.22 % |

The "spatial gate" block is therefore *not* a radius problem: only 2.2% of hotspots sit ≥15 km from every industrial/mining/refinery/oil-gas site in the cache (see §7), and merging forest/agriculture + relaxing thermal gates would re-enable roughly 200+ rows. **Radius contributes, but it is cause #4 or #5 on the list, not the root cause.**

---

## 2. SPATIAL ARCHITECTURE MAP (as-built)

```
FIRMS VIIRS CSV (1,268 hotspots)                 OSM caches (pre-fetched per state)
   latitude, longitude, frp, bright_ti4/5,            osm_industrial_sites.json      (20,231 sites)
   confidence {h,n,l}, daynight {D,N}                 osm_forest_agriculture.json    ( 3,791 sites: 3,083 forest, 708 agriculture)
        │                                                      │
        ▼                                                      ▼
 scripts/build_real_dataset.py  ────────────►  _compute_distances_from_cache(lat, lon, sites, radius_m=15000)
   • loads ONLY industrial cache  ◄──BUG──     • bbox gate: lat/lon within ±(15000/111000)×1.5 ≈ ±0.2027°  (~±22.5 km lat)
                                               • haversine nearest-distance (meters) per category
                                               • sentinel = 999000.0 m for any category with no hit
   → features: distance_to_{category}_m (7), dist_{...} (km), has_{...}5km / has_industrial_2km flags,
                count_ind_5km / count_ref_5km (bucket codes), data_sources tracked internally (not persisted)
        ▼
 labeled training set (weak supervision)  ─►  labeling_functions.py (13 LFs)  ─►  votes → consensus label
        ▼
 dataset_builder.py → training_dataset.csv (ML features)     hotspot_service.py (LIVE path, per-hotspot)
        │                                                       └─ find_nearby_geographic_objects(radius_meters=15000)
        ▼                                                          └─ compute_geospatial_context(radius_meters=15000)
 XGBoost hybrid engine (rules vote + ML vote) ─────────────────────► LIVE FEATURES DIVERGE (see §11.5)
        ▼
 run_pipeline.py → classified_hotspots_v2_enriched.csv
```

The OLTP (live) path and the OLAP (batch) path share the same *distance math* (`app/geo/distance.py`) but **different sentinels** (45,000 m live vs 999,000 m batch) and **different count features** (live hardcodes 0; batch computes buckets).

---

## 3. COMPLETE SPATIAL FEATURE INVENTORY

Full table in `reports/audit/spatial_feature_inventory.csv` (18 features). Highlights from `spatial_feature_statistics.csv`:

| Feature                     | Metric      | Status                     |
|-----------------------------|-------------|----------------------------|
| `distance_to_industry_m`    | has 175 real values (13.8%), range 60–999000 m | HEALTHY |
| `distance_to_refinery_m`    | 30 real values (2.4%)     | sparse (22 cache sites)    |
| `distance_to_oil_gas_m`     | 33 real values (2.6%)     | sparse (15 cache sites)    |
| `distance_to_mining_m`      | 165 real values (13.0%)   | sparse (94 cache sites)    |
| `distance_to_power_plant_m` | 169 real values (13.3%)   | HEALTHY, but no LF uses it |
| **`distance_to_forest_m`**  | **CONSTANT 999000, nunique=1, std=0** | **BROKEN (cache never merged)** |
| **`distance_to_agriculture_m`** | **CONSTANT 999000, nunique=1, std=0** | **BROKEN (cache never merged)** |
| `has_industrial_2km`        | 7.65% true                | matches LF industry 2 km gate |
| `has_refinery_5km` / `has_powerplant_5km` / `has_factory_5km` | 1.8% / 6.5% / 8.9% | HEALTHY |
| **`has_forest_5km` / `has_agriculture_5km`** | **CONSTANT 0 (all 1268)** | **BROKEN** |
| `count_ind_5km`             | values {0, 1, 3}          | **misnamed:** 3=≤2 km, 1=≤5 km, 0=outside; NOT a count |
| `count_ref_5km`             | values {0, 2}             | **misnamed:** 2=≤3 km, 0=outside; NOT a count |

Verified ML-vs-geo consistency: `dist_*` (km) = `distance_to_*_m` / 1000 exactly (max diff 0.0 km).
Verified ML columns: `dist_factory`/`dist_industrial_zone` are both populated from the nearest industrial-site distance (deduped, not two real features).

---

## 4. CURRENT SEARCH RADII (inventory)

| Layer                      | Value        | Where defined | Notes |
|----------------------------|--------------|---------------|-------|
| OSM discovery radius (live) | **15,000 m** | `osm_service.find_nearby_geographic_objects(radius_meters=15000)` | longest edge of bbox gate |
| Batch bbox gate            | **15,000 m ×1.5 ≈ 0.2027°** | `build_real_dataset._compute_distances_from_cache` | ±22.5 km lat, ±~21 km lon (corners ~32 km) |
| Default service radius     | 15,000 m     | `constants.DEFAULT_RADIUS_METERS` | matches discovery |
| Spatial context default    | 15,000 m     | `spatial_context.compute_geospatial_context(radius_meters=15000)` | |
| has_industrial_2km         | 2,000 m      | hard-coded in `build_real_dataset.py` | |
| has_{refinery,powerplant,factory,forest,agriculture}_5km | 5,000 m | hard-coded | |
| count_ind_5km buckets      | 2,000 / 5,000 m | hard-coded | 3=≤2 km, 1=≤5 km |
| count_ref_5km buckets      | 3,000 / 5,000 m | hard-coded | 2=≤3 km |
| LF industry proximity      | **2,000 m**  | `constants.INDUSTRIAL_PROXIMITY_M` → `THRESHOLD_INDUSTRY_PROXIMITY_M` | docstring says "1,000 m" — **doc/code mismatch** |
| LF refinery proximity      | 2,000 m      | `THRESHOLD_REFINERY_PROXIMITY_M` | |
| LF oil/gas proximity       | 2,000 m      | `THRESHOLD_OIL_GAS_PROXIMITY_M` | |
| LF mining proximity        | 2,000 m      | `THRESHOLD_MINING_PROXIMITY_M` | high-confidence LF: 3,000 m |
| LF agriculture proximity   | **15,000 m** | `THRESHOLD_AGRICULTURE_PROXIMITY_M` (hard-coded in LF, not constants) | |
| LF forest proximity        | **15,000 m** | `THRESHOLD_FOREST_PROXIMITY_M` (hard-coded in LF, not constants) | |
| LF isolation from industry | 1,500 m      | `THRESHOLD_ISOLATED_FROM_INDUSTRY_M` | used by vegetation LFs |
| Deep isolation             | 3,000 m      | `THRESHOLD_DEEP_ISOLATION_INDUSTRY_M` | |
| `POWER_PLANT_PROXIMITY_M`  | **5,000 m**  | `constants.POWER_PLANT_PROXIMITY_M` | **never consumed by any LF** (ML feature + flag only) |
| Sentinel (batch)           | 999,000 m    | `build_real_dataset.SENTINEL_DISTANCE_M` (999 km) | matches dataset_builder `MISSING_SENTINEL=999.0` km |
| Sentinel (live)            | 45,000 m     | `spatial_context.SENTINEL_DISTANCE_M` | behaviorally equivalent (both > 15 km max LF threshold) |

**Key finding:** classification is governed by the LF *evidence* thresholds (2 km industry/mining/refinery/oil-gas; 15 km forest/agriculture), **not** the 15 km discovery radius. Only hotspots 15–32 km from every cache site are dropped purely by the discovery radius; those are the "distance tail" rows (§7).

---

## 5. DATA SOURCE HEALTH

| Source | Path | Raw count | Schema / quality | Verdict |
|--------|------|----------:|------------------|---------|
| FIRMS VIIRS | `data/raw/firms_recent.csv` | 1,268 | **No `type` column** → `firms_type` always `None`. Confidence/daynight as strings (correct for `get_confidence`/`is_night`). | ⚠ `type` missing |
| Industrial OSM cache | `data/raw/osm/osm_industrial_sites.json` | 20,231 | `schema_version: "legacy-list"` vs `v3_forest_agri_mining_2026_09`; stored `category="other"` for 20,211 (stale), tags authoritative. site_type: factory 3,661, industrial_zone 14,843, power_plant 1,668, refinery 22, oil_gas 15, mining 20, volcano 2. | ⚠ schema-version drift; `_get_cache_schema_version` defined but unused |
| Forest/agri OSM cache | `data/raw/osm/osm_forest_agriculture.json` | 3,791 | 3,083 forest + 708 agriculture. **Exists and is classifiable but is never read by `build_real_dataset`.** | ❌ **not merged** |
| Site category balance (whole India) | — | — | refinery 22, oil_gas 15, mining 94 raw (mining inflates to 165 cache-visible hotspots because mining sites cluster near industrial belts) | ⚠ structurally rare |

---

## 6. FEATURE ACTIVATION (per LF, read-only run)

From `labeling_function_spatial_activation.csv` and `geospatial_audit_sim.py`:

| LF | Fired | Fire rate | Dependency chain | Verdict |
|----|------:|----------:|------------------|---------|
| lf_factory_proximity_thermal | 45 | 3.55 % | industry ≤2 km + brightness ≥325 K + conf≠low | top contributor |
| lf_nighttime_process_heat | 30 | 2.37 % | night + industry ≤2 km + FRP ≥5 MW | healthy |
| lf_industry_high_frp | 25 | 1.97 % | industry ≤2 km + FRP ≥15 MW | healthy |
| lf_industrial_zone_cluster | 25 | 1.97 % | industry ≤4 m* + FRP ≥5 MW (*density gate is a no-op) | see §11.6 |
| lf_refinery_flare | 12 | 0.95 % | refinery ≤2 km + night/persist ≥0.3/firms_type | sparse infra |
| lf_oil_gas_flare | 5 | 0.39 % | oil_gas ≤2 km + night/firms_type==3/FRP ≥5 | sparse infra |
| lf_mining_thermal_activity | 3 | 0.24 % | mining ≤2 km + FRP/brightness + conf | sparse infra |
| lf_mining_high_confidence | 0 | 0.00 % | mining ≤3 km + conf high + FRP ≥15 + isolation ≥1.5 km | compounding gates |
| lf_agriculture_vegetation_fire | 0 | 0.00 % | agri ≤15 km + veg + FRP 5–45 | ❌ cache-merge bug + firms_type |
| lf_agriculture_burn_context | 0 | 0.00 % | agri ≤15 km + FRP 5–45 + brightness | ❌ cache-merge bug |
| lf_forest_vegetation_fire | 0 | 0.00 % | forest ≤15 km + is_veg + isolation + FRP ≥5 | ❌ cache-merge bug |
| lf_strong_forest_fire | 0 | 0.00 % | forest ≤15 km + isolation + FRP ≥15 | ❌ cache-merge bug (+ FRP) |
| lf_static_industrial_heat | 0 | 0.00 % | firms_type==3 + industry ≤4 km | ❌ firms_type never available |

**Merging the forest/agriculture cache (simulation, in-memory only) changes this to:**
forest vegetation 6, agriculture vegetation 1, agriculture burn 1 → consensus labels shift from 1,208 unclassified to **1,204** (adds 3 `forest_natural_fire`, 1 `agricultural_burn`).

Why so little, despite 177 forest-proximate hotspots? — see §7 decay:
- 177 within 15 km → **6 have FRP ≥5 MW** (96.6% fail the thermal gate) → 6 forest LF fires.
- 34 within 15 km farmland → 12 have FRP 5–45 MW → **1 isolated from industry** (isolation gate) → 1 agriculture LF fire.

---

## 7. RADIUS SENSITIVITY (true nearest distances, recomputed from raw caches)

Full table in `radius_sensitivity_report.csv` (categories × 100 m–50 km). Activation as a function of threshold, **% of 1,268 hotspots with an entity inside the given radius**:

| Category | @2 km | @5 km | @10 km | @15 km | @30 km | cache sites in India |
|----------|------:|------:|-------:|-------:|-------:|---------------------:|
| industry | 7.65 % | 8.91 % | 9.54 % | ~10 % | 14.35 % | 17,468+ |
| power_plant | 4.65 % | 6.55 % | 7.97 % | ~9.5 % | 13.80 % | 1,668 |
| forest | — | 5.05 % | 11.59 % | **13.96 %** | 24.53 % | 3,083 |
| mining | 0.55 % | 1.89 % | 5.84 % | ~8 % | 13.25 % | 94 |
| agriculture | 0 % | 2.13 % | 2.68 % | ~2.9 % | 8.20 % | 708 |
| refinery | 1.18 % | 1.81 % | 1.81 % | ~2 % | 2.60 % | 22 |
| oil_gas | 0.39 % | 1.58 % | 2.52 % | ~2.5 % | 2.60 % | 15 |

**Production-bbox recomputation** (same 0.2027° gate the builder actually uses, applied to raw caches):
industry **175** (13.8 %), refinery 30 (2.4 %), oil_gas 33 (2.6 %), mining 165 (13.0 %), power_plant 169 (13.3 %), **forest 282 (22.2 %)**, **agriculture 84 (6.6 %)**.

Three conclusions:
1. **The discovery radius is NOT the limiter.** Under the production bbox, industry/refinery/oil_gas/mining/power_plant all drop to the same hotspot sets that the builder stored (verified §10). Increasing discovery radius 15→30 km adds at most a few percentage points per category; shrinking it to 5 km would lose only ~1 % of industrial rows but would cut forest/agriculture opportunity by ~60 %.
2. **Forest is a genuinely recoverable class — at the 15 km LF threshold it would reach 14 % of rows** if the cache were merged. The agriculture LF would reach ~3 %. This is the single largest *config-throughput* lever identified, ahead of any radius change.
3. **Mining/oil_gas/refinery are infra-bound, not radius-bound.** Their site counts (94/15/22 for all of India) cap attainable activation at ~14 %/{2.5 %}/{2.6 %} even at 50 km. No threshold change can create entities that do not exist in OSM.

---

## 8. PROXIMITY vs CONTAINMENT

- **No polygon containment anywhere.** Grep for `intersects|contains|within|buffer|shapely|geopandas|Polygon` across the entire codebase: **zero geometric predicates**. All 7 categories are treated as dimensionless points (`lat`/`lon`) and matched by nearest-center haversine.
- **Forest/agriculture are proximity-flags, not land-cover containment.** The LF names imply vegetation context (`forest`/`agriculture` landuse points), but an active fire is never tested for whether it falls *inside* the forest polygon — only whether the nearest forest point is ≤15 km. This inflates false-vegetation risk in the 15-km band near industrial/farmland edges (the isolation gate at 1.5 km is the only mitigation).
- No buffered polygon distance, no area weighting, no land-cover intersection. For a fire-detection system this is the design gap that polygons would have fixed; it is currently a *quality* limitation, not the imbalance's cause.

---

## 9. LF IMPACT (spatial contribution to labels)

- Current weak-supervision distribution: `industrial_fire` 49 (3.9 %), `gas_flare` 8 (0.6 %), `mining_activity` 3 (0.2 %), **`unclassified` 1,208 (95.2 %)**.
- 4 of 13 LFs produce **every** assigned label (industry/refinery/oil_gas/night heat). 4 more (mining_high_confidence, both agriculture, both forest, static heat) produce **zero**.
- **57.1 % of LF capacity is inert** (0 fires) — 8 of 13 LFs never vote. Of those, 7 are INFO-structure for classes that *do* exist in the raw data.
- `lf_industrial_zone_cluster` "density" gate (`count_ind_5km >= 2`) is **effectively a no-op**: value 3 (≤2 km) is what satisfies it, but the LF separately requires `dist_ind ≤ 2000 m`, making the bucket test redundant. The cluster semantics the LF documents (multiple industrial sites) are not actually measured.

---

## 10. DISTANCE MATH / UNITS / COORDINATES (verification)

Stored vs recomputed haversine (nearest-site, meters) across all 1,268 rows:

| Category | stored "found" rows | recomputed | max \|Δ\| | verdict |
|----------|--------------------:|-----------:|----------:|---------|
| industry | 175 | 175 | 0.1 m | ✓ (rounding only) |
| refinery | 30 | 30 | 0.1 m | ✓ |
| oil_gas | 33 | 33 | 0.1 m | ✓ |
| mining | 165 | 165 | 0.1 m | ✓ |
| power_plant | 169 | 169 | 0.1 m | ✓ |
| forest | 0 | 282 (would-be) | — | ❌ never computed |
| agriculture | 0 | 84 (would-be) | — | ❌ never computed |

- **Units:** haversine returns meters; `dist_*` km columns are exact /1000; `get_distance_meters` prioritizes `_m` columns and converts. 1° latitude verified ≈111 km; zero-distance verified 0.
- **Coordinate order:** input is consistently `(latitude, longitude)`; swapped interpretation yields materially different (wrong) distances. OSM cache `lat/lon` and FIRMS `latitude/longitude` are in the same order.
- **Bbox gate:** matches builder's stored "found" row sets exactly for every industrial category (175=175 etc.), proving the 1.5× generosity never drops an industrial hit that the true 15 km circle would find.
- **CRS:** WGS-84 everywhere; no reprojection tuning needed.

**Distance logic: HEALTHY. No radius bug in the batch math.**

---

## 11. DEFINITE BUGS (ordered by severity)

1. **B1 — Forest/agriculture cache never merged** (`build_real_dataset.py` loads only `OSM_INDUSTRIAL_CACHE_PATH`). `dist_forest`/`dist_agriculture`/`has_forest_5km`/`has_agriculture_5km`/`distance_to_{forest,agriculture}_m` are constant. 3,791 real sites unused. Root cause B (needed data exists) + C (parsed separately but never joined).
2. **B2 — FIRMS `type` column absent** → `firms_type` is always `None`. Four LF branches designed around it (`lf_oil_gas_flare`, `lf_refinery_flare`, `lf_static_industrial_heat`, vegetation conditions in forest/agriculture LFs) fire at structural discount. Root cause B (data present in coarser VIIRS is unavailable; MODIS-only field).
3. **B3 — Live/batch feature divergence** (`hotspot_service._build_feature_record` hardcodes `count_ind_5km`, `count_ref_5km`, `count_forest_5km`, `count_agriculture_5km` = 0). The live classifier sees materially different features than Cb (batch), so live-vs-batch labels can disagree on identical inputs. Root cause E (semantic inconsistency at runtime).
4. **B4 — `count_ind_5km`/`count_ref_5km` are misnamed bucket codes, not counts.** V3 = "nearest ≤2 km", 1 = "≤5 km but >2", 0 = none (ref: 2 = "≤3 km"). Any consumer interpreting them as entity counts (including `lf_industrial_zone_cluster`'s "dense zone" gate) is reading noise. Root cause D (semantics vs name), severity lower because the industry LF's density gate is redundant anyway.
5. **B5 — Sentinels duplicated with different magic numbers** (999,000 batch vs 45,000 live) and duplicated 45,000 literals inside LFs (`dist_ind >= 45000.0`). Behaviorally equivalent today (§4 test) but fragile; 999 km also leaks into ML distance features as a large, non-informative constant vs the live 45 km.
6. **B6 — `lf_industrial_zone_cluster` density gate is a no-op** (see §9). The LF does not measure what its label branch claims.
7. **B7 — OSM cache schema-version drift:** `legacy-list` vs the version string in code (`v3_forest_agri_mining_2026_09`); `_get_cache_schema_version` is defined but never called, so stale caches can never force a re-query. Combined with B1 this is how the forest/agri data was (mis)organized.
8. **B8 — Docstring/constant mismatch:** `INDUSTRIAL_PROXIMITY_M` (2,000 m) is documented as 1,000 m in LF docstrings. Cosmetic, but a real risk for future calibration work.

---

## 12. RECOMMENDED NEXT FIXES (priority order — implementation deferred)

### Priority 0 — Highest (fix the disabled data, not the radii)
- **P0.1 Merge `osm_forest_agriculture.json` into `_compute_distances_from_cache`** the same way as the industrial cache (bbox gate + haversine + classify with category fallback). No threshold change. Expected effect: `dist_forest`/`dist_agriculture`/flags become non-constant; ~211 previously-frozen rows gain spatial evidence; forest/agri LFs fire for a handful of high-FRP rows. Re-run the audit-sim to confirm it produces non-constant features.
- **P0.2 Add `firms_type` (or a MODIS overlay) into the FIRMS ingestion**, OR explicitly document VIIRS-`type` unavailability and delete the `firms_type`-gated branches so LFs don't silently under-vote. This is the second-largest evidence lever.

### Priority 1 — High (make the pipeline honest)
- **P1.1 Remove live/batch divergence** (B3): compute `count_*` in `_build_feature_record` from the service's own spatial result instead of hardcoding 0, and track sentinel + data-source per feature.
- **P1.2 Single sentinel constant** (B5): one `SENTINEL_DISTANCE_M` consumed by batch, live, and LFs; remove literal 45000.0 from LF bodies.
- **P1.3 Fix `count_*_5km` semantics** (B4): rename to `bucket_ind_2km` / `bucket_ind_5km` style or replace with a real neighbor count from `compute_geospatial_context` (it already has a neighbor list).

### Priority 2 — Medium (measure then calibrate; NO radius change until evidence demands)
- **P2.1 Calibrate the industry 2 km evidence threshold against a controlled A/B** (e.g., 1.5/2.0/2.5/3.0 km) using precision-on-consensus, not ad-hoc doc rotation. The current 2 km captures 7.65 % of rows; the boundary is plausibly good, but there is no calibration record.
- **P2.2 Forest/agriculture 15 km**: keep 15 km (matches the satellite buffer design); revisit only after P0.1 so the class is measured *with* data, not against constants.
- **P2.3 Mining 2 km vs its 10 km band**: mining is genuinely sparse; a 2→10 km (LF) experiment is the single largest *potential* radius-driven label increase (0.55 % → 5.8 %) and must be validated for error-rate before changing.

### Priority 3 — Low
- **P3.1** Wire `_get_cache_schema_version` to force cache re-fetch on schema mismatch (B7).
- **P3.2** Fix LF docstrings (B8) and make agriculture/forest thresholds constants (they are the only LF thresholds not in `constants.py`).
- **P3.3** Consider polygon containment for forest/farmland as a *future* feature (post-merge), replacing the 15 km proximity assumption for label semantics.

---

## 13. WHAT NOT TO CHANGE (yet)

- **Do NOT raise the 5 km/10 km disclosure flags to "solve" imbalance.** Evidence flags are feature/logic, not policy: inflating `has_*_5km` bands until flags spike cheapens the features before the cache-merge bug is fixed.
- **Do NOT add SMOTE / synthetic oversampling / class weights to the model** while B1/B2 are open — you would be resampling noise (constant features + missing firms_type). Reoxygenate the data first.
- **Do NOT raise the discovery radius (15 km) defensively.** It is not the binding constraint (§7). Wasteful live API cost for ≤ few %.
- **Do NOT widen the industry 2 km LF gate pre-calibration** — it is the most reliable vote source; protect its precision.
- **Do NOT rely on "unclassified" as an evidence-driven class** until B0-row features carry real spatial variance.

---

## 14. TEST RESULTS

- Full suite: **53 passed, 0 failed** (`python -m pytest -v`) in ~5.3 s.
- New **19-test `tests/test_geospatial_audit.py`** guard regressions:
  - Distance units (m), ~111 km/1°, zero-distance, coordinate-order, `get_distance_meters` priority.
  - Radius boundary flip (2 km LF gate, 5 km flags), sentinel-vs-flag semantics.
  - Producer contract: entity inside/outside search radius, **empty-cache detectability**, query-hit vs query-miss distinguishability.
  - Config consistency: LF thresholds == constants, power-plant radius unused (locked), search radius == builder constant, **sentinel behavioral equivalence** (999 km ≡ 45 km both > 15 km).
  - **Data guards:** forest/agri cache contains >1,000 forest + >200 agriculture sites AND `dist_forest`/`dist_agriculture` are constant (999.0) in the current canonical build — the bug is pinned until a merge makes `nunique > 1`; industry distances have real variance (contrast).
- Prior suites (34 tests: data lineage, build encoding, pipeline CLI) remain green.

---

## 15. SCIENTIFIC CONCLUSION — EXECUTIVE ROOT-CAUSE MATRIX

| # | Root cause | Result in data | Radius involved? | Weight for `unclassified` |
|---|-----------|----------------|------------------|---------------------------|
| **A** | No source data / platform has none | oil/gas (15), refinery (22), mining (94) sites for all of India | Discovery radius WOULD pass them; infra is few | **Contributors: refinery/oil-gas/mining ~3 %** — not the imbalance's main driver |
| **B** | Data exists but never retrieved/joined | forest/agri cache (3,083/708) separate, **never merged**; `firms_type` column absent | No | **★ CORE #1 — kills 4 LFs + 2 entire classes** |
| **C** | Data retrieved but parsed wrong | cache `category` stale ("other" ×20,211); tags reclassify correctly at build (so mitigated) | No | Low (mitigated by tag reclassification) |
| **D** | Distance computed wrong | **NOT FOUND** — haversine verified to 0.1 m, units/CRS/order correct | Yes (if wrong) | **None — distance logic is sound** |
| **E** | Threshold too restrictive | industry 2 km = 7.65 %, mining 2 km = 0.55 %, forest 15 km = 13.96 % (post-merge); thermal FRP ≥5 MW blocks 65.1 % of all hotspots | **Yes — EVIDENCE thresholds** | **★ Contributor — but thermal gates (not radii) dominate; radius effect ~4–5 %** |
| **F** | Relationship conceptually wrong | `count_*` mislabeled buckets; proximity-vs-containment; density no-op; live/batch divergence | Indirect | Real but secondary; deflates precision, not volume |
| **G** | Data genuinely lacks evidence | only ~5–9 % of hotspots have high-FRP/night signatures near *any* industrial site; OSM rural-fire coverage limited | No | **★ The honest floor — even a perfect pipeline keeps `unclassified` high (≈ single-digit % real assignments)** |

**Bottom line:** The extreme 95.2 % `unclassified` is **not primarily a search-radius problem**. In rank order it is caused by (1) the **forest/agriculture cache never being merged** (+ all four vegetation/static LFs inert) [B], (2) **`firms_type` unavailable** [B], (3) **thermal evidence gates (FRP ≥5 MW)** which filter 65 % of hotspots even with perfect spatial data [E/G], (4) **genuine OSM sparsity** for refinery/oil_gas/mining [A/G], and (5) feature-semantics defects (`count_*`, live/batch divergence) [F]. The *discovery radius* (15 km) is verified non-binding; the *evidence thresholds* (2 km industry/mining; 5 km→15 km vegetation band) are secondary contributors that should only be tuned after the data bugs are fixed and measured.

**In one sentence:** "The imbalance is real, the geometry is correct, and the fix is to merge the data that already exists on disk — then, and only then, re-measure whether any threshold should move."