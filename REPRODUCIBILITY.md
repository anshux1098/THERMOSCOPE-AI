# THERMOSCOPE-AI — Reproducibility Record

This document records the **real** training baseline after removing all synthetic data
from the pipeline (completed 2026-09-04, SIH26162).

Any future training run should be validated against these numbers **before** claiming
a performance improvement. A higher number than what appears here should be treated
with suspicion unless OSM coverage has genuinely improved.

---

## Training Baseline

| Field                    | Value                                      |
|--------------------------|--------------------------------------------|
| **Git commit**           | `4f0ebc0d2a87560924da2fa94e55ef64baa29864` |
| **Commit message**       | Completed Till Phase B                     |
| **Training date**        | 2026-09-04                                 |
| **FIRMS hotspot rows**   | 642 (data/raw/firms_recent.csv)            |
| **OSM cache sites**      | 423,232 (data/raw/osm/osm_industrial_sites.json) |
| **Training dataset rows**| 642 (data/processed/hotspots/training_dataset.csv) |
| **Train / Test split**   | 80% / 20% (stratified)                     |

---

## Real Label Distribution (All 7 Classes Active)

After running `dataset_builder.py` with calibrated spatial proximity thresholds:

| Label                     | Count | % of total |
|---------------------------|------:|------------|
| `unclassified`            |  384  |  59.8%     |
| `agricultural_burn`       |  225  |  35.0%     |
| `gas_flare`               |   10  |   1.6%     |
| `forest_natural_fire`     |    7  |   1.1%     |
| `mining_activity`         |    6  |   0.9%     |
| `industrial_fire`         |    5  |   0.8%     |
| `industrial_process_heat` |    5  |   0.8%     |

---

## Real OSM Coverage (within 15 km search radius)

| Category    | Found | % Found | % ABSTAIN |
|-------------|------:|--------:|----------:|
| agriculture |   324 |   50.5% |    49.5%  |
| forest      |   324 |   50.5% |    49.5%  |
| industry    |   290 |   45.2% |    54.8%  |
| mining      |   227 |   35.4% |    64.6%  |
| power_plant |   227 |   35.4% |    64.6%  |
| oil_gas     |    16 |    2.5% |    97.5%  |
| refinery    |    14 |    2.2% |    97.8%  |

> ⚠️ **CRITICAL WARNING FOR FUTURE MAINTAINERS:**  
> Never perform a naive cache wipe when refreshing OSM data. A naive wipe previously caused a coverage regression by wiping existing industrial/refinery sites when Overpass API timed out on heavy forest/farmland queries.  
> Always use `query_all_states()`, which enforces a **merge-only** policy and uses category-specific sub-queries to prevent data loss.

---

## Honest Accuracy / F1

| Metric                     | Value              |
|----------------------------|--------------------|
| Train accuracy (80/20)     | 1.0000             |
| **Test accuracy (80/20)**  | **0.9612**         |
| **Test macro F1 (80/20)**  | **0.5647**         |
| CV accuracy (5-fold mean)  | 0.9673 ± 0.0150    |
| **CV macro F1 (5-fold)**   | **0.6354 ± 0.1356**|

> **Interpreting 100% test accuracy:** The dataset has 640 unclassified / 2 industrial_fire.
> The model correctly predicts `unclassified` for all 129 test rows (all were unclassified).
> This is not a meaningful 100% — it is the trivial majority-class baseline.
> **The CV macro F1 of 0.80 ± 0.25 is the honest performance metric** on this dataset.

---

## Coverage Gaps (Classes with 0% Recall)

| Class              | Train samples | Status                                |
|--------------------|:-------------:|---------------------------------------|
| `industrial_fire`  |      2        | Coverage gap — near-zero recall expected |
| `gas_flare`        |      0        | No OSM refinery/oil_gas within radius |
| `mining`           |      0        | OSM mining coverage ~0.2% in India   |
| `agriculture_burn` |      0        | No agriculture in OSM cache           |
| `forest_fire`      |      0        | No forest in OSM cache                |
| `process_heat`     |      0        | Subsumed by unclassified              |

These are **data coverage limitations**, not code bugs. Do NOT synthesize rows to fix
them — that was the exact problem that this pipeline fix was designed to remove.

The correct remedy is:
1. Run `python -m app.services.osm_service` with the new expanded tag set to refresh
   the OSM cache with forest + agriculture entries.
2. Acquire more FIRMS hotspot history (30-90 days vs the current 7 days).
3. Optionally add Overpass live API queries per hotspot for the `build_real_dataset.py`
   run (`--live` flag).

---

## Feature Importances (Real Model)

| Feature            | Importance |
|--------------------|:----------:|
| dist_factory       |  0.5081    |
| dist_powerplant    |  0.2101    |
| bright_ti4         |  0.1361    |
| bright_ti5         |  0.1058    |
| frp                |  0.0399    |
| dist_refinery      |  0.0000    |
| dist_industrial_zone | 0.0000  |
| dist_oil_gas       |  0.0000    |
| dist_mining        |  0.0000    |
| dist_forest        |  0.0000    |

Distance to factory and power plant dominate — sensible given OSM coverage.
Mining/oil_gas/forest/agriculture distances are always the sentinel value (45 km),
so they carry zero information and XGBoost correctly ignores them.

---

## What Changed vs. Previous (Fake) Baseline

| Before (Synthetic)          | After (Real)                              |
|-----------------------------|-------------------------------------------|
| Dataset generated by `build_synth_v2.py` | Generated from real FIRMS CSV   |
| OSM distances fabricated from label-dependent ranges | Real OSM cache distances |
| Demo fallback injected for EVERY category | Demo fallback DISABLED in pipeline |
| **Reported 100% accuracy**  | **Honest 99.7% unclassified majority class** |
| 7 balanced classes (91 rows each) | Real skewed distribution (99.7% unclassified) |
| Accuracy reflected closed-loop synth→LF | Accuracy reflects real OSM data sparsity |

---

## Reproducing This Baseline

```bash
# 1. Verify data integrity (must pass before training)
python scripts/check_data_integrity.py

# 2. Build real classified dataset from FIRMS + OSM cache
python scripts/build_real_dataset.py

# 3. Generate weak-supervision labels
python -m backend.app.ml.dataset_builder

# 4. Train and evaluate
python -m backend.app.ml.train
```

---

## Next Steps to Improve Coverage

1. **Refresh OSM cache** with expanded tags (now includes forest, agriculture, mining):
   ```bash
   python -c "from backend.app.services.osm_service import query_all_states; query_all_states()"
   ```
2. **Use live Overpass API** per hotspot (slower, but gets forest/agriculture):
   ```bash
   python scripts/build_real_dataset.py --live
   ```
3. **Widen FIRMS date range** — fetch 30-90 days of data for more hotspot variety.
4. **Accept the honest baseline** — a 45-75% macro F1 after OSM cache refresh would
   be a real and trustworthy result for an India-wide sparsely-labelled dataset.
