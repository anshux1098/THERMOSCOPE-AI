import { ClassKey, CLASS_DISPLAY_NAMES, Hotspot } from '../types/hotspot';

const API_BASE = 'http://127.0.0.1:8000';

/** Backend presentation shape (Phase 3 batch endpoint). Loose on purpose. */
export interface PresentationResponse {
  hotspot?: Record<string, unknown>;
  spatial_context?: Record<string, unknown>;
  classification?: Record<string, unknown>;
  [key: string]: unknown;
}

function str(v: unknown, fallback = ''): string {
  return typeof v === 'string' ? v : fallback;
}

function num(v: unknown, fallback = 0): number {
  const n = typeof v === 'number' ? v : parseFloat(String(v ?? ''));
  return Number.isFinite(n) ? (n as number) : fallback;
}

const VALID: ClassKey[] = [
  'industrial_fire',
  'gas_flare',
  'mining_activity',
  'agricultural_burn',
  'forest_natural_fire',
  'industrial_process_heat',
  'unclassified',
];

/** Map backend presentation fields onto the dashboard's internal Hotspot type. */
export function presentationToHotspot(p: PresentationResponse, index: number): Hotspot {
  const h = (p.hotspot ?? {}) as Record<string, unknown>;
  const s = (p.spatial_context ?? {}) as Record<string, unknown>;
  const c = (p.classification ?? {}) as Record<string, unknown>;
  const cls = VALID.includes(c['final_label'] as ClassKey)
    ? (c['final_label'] as ClassKey)
    : 'unclassified';
  const m = (v: unknown): number | null => {
    const n = num(v, NaN);
    return Number.isFinite(n) && n < 45000 ? Math.round(n) : null;
  };
  return {
    id: `HS-API-${String(index + 1).padStart(4, '0')}`,
    lat: num(h['latitude']),
    lon: num(h['longitude']),
    frp: num(h['frp']),
    bright_ti4: num(h['bright_ti4'] ?? h['brightness']),
    confidence: 'nominal',
    daynight: 'D',
    acq_date: str(h['acq_date'], '2026-09-02'),
    satellite: str(h['satellite'], 'Unknown'),
    classification: {
      class: cls,
      display_name: CLASS_DISPLAY_NAMES[cls],
      confidence: num(c['hybrid_confidence']),
      confidence_level: 'Low',
      decision_source: str(c['decision_source'], 'uncertain'),
      risk_level: 'Medium',
      requires_human_review: true,
      review_reason: str(c['review_reason'] || null) || null,
      explanation: Array.isArray(c['explanation'])
        ? (c['explanation'] as string[])
        : [],
    },
    spatial_context: {
      industry: m(s['nearest_industry_m'] ?? s['distance_to_industry_m']),
      refinery: m(s['nearest_refinery_m'] ?? s['distance_to_refinery_m']),
      oil_gas: m(s['nearest_oil_gas_m'] ?? s['distance_to_oil_gas_m']),
      mining: m(s['nearest_mining_m'] ?? s['distance_to_mining_m']),
      forest: m(s['nearest_forest_m'] ?? s['distance_to_forest_m']),
      agriculture: m(s['nearest_agriculture_m'] ?? s['distance_to_agriculture_m']),
      power_plant: m(s['nearest_power_plant_m'] ?? s['distance_to_power_plant_m']),
    },
    rule_engine: { active_votes: 0, total_lfs: 14, active_lfs: [] },
    ml_engine: { predicted_class: cls, probabilities: { [cls]: 1 } },
  };
}

export async function healthCheck(timeoutMs = 4000): Promise<boolean> {
  const ctrl = new AbortController();
  const t = window.setTimeout(() => ctrl.abort(), timeoutMs);
  try {
    const res = await fetch(`${API_BASE}/health`, { signal: ctrl.signal });
    if (!res.ok) return false;
    const data = await res.json();
    return data?.status === 'ok';
  } catch {
    return false;
  } finally {
    window.clearTimeout(t);
  }
}
