import {
  ALL_LF_NAMES,
  ClassKey,
  CLASS_DISPLAY_NAMES,
  Hotspot,
} from '../types/hotspot';

/**
 * The enriched CSV carries vote counts but no per-LF names. The backend's
 * own explanation text names the voting LFs, so recover them from there.
 * Only names from the known registry are accepted — nothing is guessed.
 */
function extractActiveLFs(bullets: string[], rulePrediction: string): { name: string; vote: string }[] {
  const text = bullets.join(' ');
  const found: { name: string; vote: string }[] = [];
  for (const name of ALL_LF_NAMES) {
    if (text.includes(name) && !found.some((f) => f.name === name)) {
      found.push({ name, vote: rulePrediction });
    }
  }
  return found;
}

const VALID_CLASSES: ClassKey[] = [
  'industrial_fire',
  'gas_flare',
  'mining_activity',
  'agricultural_burn',
  'forest_natural_fire',
  'industrial_process_heat',
  'unclassified',
];

function toClass(v: string): ClassKey {
  return (VALID_CLASSES as string[]).includes(v) ? (v as ClassKey) : 'unclassified';
}

function num(v: string, fallback = 0): number {
  const n = parseFloat(v);
  return Number.isFinite(n) ? n : fallback;
}

/** km value from CSV -> meters for internal type; 999 sentinel -> null. */
function kmToM(v: string): number | null {
  const n = parseFloat(v);
  if (!Number.isFinite(n) || n >= 999) return null;
  return Math.round(n * 1000);
}

function bool(v: string): boolean {
  return v.trim().toLowerCase() === 'true';
}

function capLevel(v: string): 'High' | 'Medium' | 'Low' {
  const s = v.trim().toLowerCase();
  if (s === 'high') return 'High';
  if (s === 'medium') return 'Medium';
  return 'Low';
}

function riskFromScore(v: string): string {
  const n = parseFloat(v);
  if (!Number.isFinite(n)) return 'Low';
  if (n >= 75) return 'Critical';
  if (n >= 50) return 'High';
  if (n >= 25) return 'Medium';
  return 'Low';
}

/** Parse Python-list-encoded bullets: ["a", "b"] with "" escapes. */
function parseBullets(raw: string): string[] {
  const t = raw.trim();
  if (!t) return [];
  if (t.startsWith('[') && t.endsWith(']')) {
    const inner = t.slice(1, -1);
    const parts = inner.split('","');
    return parts
      .map((p) => {
        let s = p.trim();
        if (s.startsWith('"')) s = s.slice(1);
        if (s.endsWith('"')) s = s.slice(0, -1);
        return s.replace(/""/g, '"').trim();
      })
      .filter((s) => s.length > 0);
  }
  return [t];
}

/** Minimal RFC4180-ish row splitter: respects "..." quoting, "" escape. */
function splitRow(line: string): string[] {
  const out: string[] = [];
  let cur = '';
  let inQuotes = false;
  for (let i = 0; i < line.length; i++) {
    const c = line[i];
    if (inQuotes) {
      if (c === '"') {
        if (line[i + 1] === '"') {
          cur += '"';
          i++;
        } else {
          inQuotes = false;
        }
      } else {
        cur += c;
      }
    } else if (c === '"') {
      inQuotes = true;
    } else if (c === ',') {
      out.push(cur);
      cur = '';
    } else {
      cur += c;
    }
  }
  out.push(cur);
  return out.map((v) => v.trim());
}

export function parseCSV(text: string): Hotspot[] {
  const lines = text.split('\n').filter((l) => l.trim());
  if (lines.length < 2) return [];
  const headers = splitRow(lines[0]);
  const out: Hotspot[] = [];
  for (let r = 1; r < lines.length; r++) {
    try {
      const values = splitRow(lines[r]);
      const obj: Record<string, string> = {};
      headers.forEach((h, i) => {
        obj[h] = values[i] ?? '';
      });
      out.push(convertRowToHotspot(obj, out.length));
    } catch {
      /* skip malformed rows */
    }
  }
  return out;
}

export function convertRowToHotspot(row: Record<string, string>, index: number): Hotspot {
  const cls = toClass(row['final_label'] || 'unclassified');
  const mlPred = toClass(row['ml_prediction'] || 'unclassified');
  const topP = num(row['ml_top_probability'], 0);
  const probs: Record<string, number> = {};
  for (const c of VALID_CLASSES) probs[c] = c === mlPred ? topP : 0;

  const needsReview = row['requires_human_review']
    ? bool(row['requires_human_review'])
    : cls === 'unclassified';

  const bullets = parseBullets(row['explanation_bullets'] || '');
  const rulePrediction = toClass(row['rule_prediction'] || 'unclassified');
  const activeVotes = Math.max(0, parseInt(row['rule_active_votes'] || '0', 10) || 0);

  return {
    id: `HS-R-${String(index + 1).padStart(4, '0')}`,
    lat: num(row['latitude']),
    lon: num(row['longitude']),
    frp: num(row['frp']),
    bright_ti4: num(row['bright_ti4']),
    confidence: 'nominal',
    daynight: 'D',
    acq_date: row['acq_date'] || '2026-09-02',
    satellite: 'Unknown',
    classification: {
      class: cls,
      display_name: CLASS_DISPLAY_NAMES[cls],
      confidence: num(row['hybrid_confidence'], 0),
      confidence_level: capLevel(row['confidence_level'] || 'low'),
      decision_source: row['decision_source'] || 'uncertain',
      risk_level: riskFromScore(row['risk_score']),
      requires_human_review: needsReview,
      review_reason: row['review_reason'] || null,
      explanation: bullets,
      conflict: row['conflict'] ? bool(row['conflict']) : false,
    },
    spatial_context: {
      industry: kmToM(row['dist_industry_km']),
      refinery: kmToM(row['dist_refinery_km']),
      oil_gas: kmToM(row['dist_oil_gas_km']),
      mining: kmToM(row['dist_mining_km']),
      forest: kmToM(row['dist_forest_km']),
      agriculture: kmToM(row['dist_agriculture_km']),
      power_plant: kmToM(row['dist_powerplant_km']),
    },
    rule_engine: {
      active_votes: activeVotes,
      total_lfs: 14,
      active_lfs: extractActiveLFs(bullets, rulePrediction),
    },
    ml_engine: {
      predicted_class: mlPred,
      probabilities: probs,
    },
  };
}
