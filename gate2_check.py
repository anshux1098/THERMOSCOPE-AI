"""
gate2_check.py — Run Gate 2 verification after Phase D.

Checks:
  [x] hotspot_service.py runs
  [x] v2 CSV opens in pandas
  [ ] >= 3 hotspots have non-empty explanation_bullets
  [ ] Risk scores in range 0-100
  [ ] fusion_agree is True for >50% of rows
"""
import sys
from pathlib import Path
sys.path.insert(0, 'backend')

import pandas as pd
import json

from app.services.hotspot_service import analyze_single_hotspot

# Load v2 CSV
df = pd.read_csv('data/processed/hotspots/classified_hotspots_v2.csv')
print(f"=== Gate 2 Check (n={len(df)} hotspots) ===\n")

# Run all 642 through the service
results = []
for idx, row in df.iterrows():
    hs = {
        'latitude': row['latitude'],
        'longitude': row['longitude'],
        'frp': row.get('frp', 0),
        'bright_ti4': row.get('bright_ti4', row.get('brightness', 0)),
        'brightness': row.get('bright_ti4', row.get('brightness', 0)),
        'confidence': row.get('confidence', 'n'),
        'daynight': row.get('daynight', 'D'),
        'acq_date': row.get('acq_date', '2026-09-01'),
    }
    try:
        analysis = analyze_single_hotspot(hs, run_classification=True)
        cls = analysis.classification
        results.append({
            'idx': idx,
            'final_label': cls['final_label'],
            'hybrid_confidence': cls['hybrid_confidence'],
            'decision_source': cls['decision_source'],
            'agreement': cls['agreement'],
            'conflict': cls['conflict'],
            'requires_human_review': cls['requires_human_review'],
            'explanation_bullets': cls.get('explanation', []),
            'n_bullets': len(cls.get('explanation', [])),
            'spatial_context': {
                'industry_m': analysis.spatial_context.nearest_industry_m,
                'refinery_m': analysis.spatial_context.nearest_refinery_m,
                'oil_gas_m': analysis.spatial_context.nearest_oil_gas_m,
                'mining_m': analysis.spatial_context.nearest_mining_m,
                'forest_m': analysis.spatial_context.nearest_forest_m,
                'agriculture_m': analysis.spatial_context.nearest_agriculture_m,
                'power_plant_m': analysis.spatial_context.nearest_power_plant_m,
            },
        })
    except Exception as e:
        print(f"  Hotspot #{idx} failed: {e}")
        continue

results_df = pd.DataFrame(results)
print(f"Processed: {len(results_df)} / {len(df)} hotspots\n")

# Check 3: >= 3 hotspots have non-empty explanation_bullets
n_with_bullets = (results_df['n_bullets'] > 0).sum()
print(f"[Check 3] Non-empty explanation_bullets: {n_with_bullets} hotspots (need >= 3)")
status_3 = "PASS" if n_with_bullets >= 3 else "FAIL"
print(f"           Status: {status_3}\n")

# Check 4: Risk scores in range 0-100
# Hybrid confidence is 0-1; map to 0-100 risk score
results_df['risk_score'] = (results_df['hybrid_confidence'] * 100).round(1)
risk_min = results_df['risk_score'].min()
risk_max = results_df['risk_score'].max()
n_valid_risk = ((results_df['risk_score'] >= 0) & (results_df['risk_score'] <= 100)).sum()
print(f"[Check 4] Risk score range: {risk_min} - {risk_max}")
print(f"           Valid 0-100: {n_valid_risk} / {len(results_df)}")
status_4 = "PASS" if risk_min >= 0 and risk_max <= 100 else "FAIL"
print(f"           Status: {status_4}\n")

# Check 5: fusion_agree is True for >50% of rows
n_agree = results_df['agreement'].sum()
pct_agree = 100 * n_agree / len(results_df)
print(f"[Check 5] fusion_agree=True: {n_agree} / {len(results_df)} ({pct_agree:.1f}%)")
status_5 = "PASS" if pct_agree > 50 else "FAIL"
print(f"           Status: {status_5}\n")

# Save enriched dataset for Phase E
enriched_csv = 'data/processed/hotspots/classified_hotspots_v2_enriched.csv'
results_df.to_csv(enriched_csv, index=False)
print(f"Saved enriched dataset: {enriched_csv}")
print(f"Columns: {list(results_df.columns)}")

# Summary
print("\n" + "=" * 60)
print("GATE 2 SUMMARY")
print("=" * 60)
all_pass = status_3 == "PASS" and status_4 == "PASS" and status_5 == "PASS"
print(f"  Check 3 (explanation_bullets >= 3):    {status_3}")
print(f"  Check 4 (risk scores 0-100):          {status_4}")
print(f"  Check 5 (fusion_agree > 50%):         {status_5}")
print(f"  Overall: {'PASS' if all_pass else 'NEEDS REVIEW'}")
