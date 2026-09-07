import { Hotspot } from '../../types/hotspot';

interface Props {
  hotspot: Hotspot;
}

/** Crude Indian region heuristic (placeholder until real reverse-geocoding). */
export function inferLocation(lat: number, lon: number): string {
  if (lat > 35 || lat < 6 || lon < 68 || lon > 97) return 'outside India';
  if (lat >= 28 && lon >= 74 && lon < 78) return 'Haryana / Delhi region';
  if (lat >= 22 && lat < 24 && lon >= 69 && lon < 73) return 'Gujarat (Jamnagar / Vadinar belt)';
  if (lat >= 18 && lat < 21 && lon >= 72 && lon < 74) return 'Maharashtra (Mumbai belt)';
  if (lat >= 30 && lat < 32 && lon >= 77 && lon < 80) return 'Uttarakhand / Himachal foothills';
  if (lat >= 21 && lat < 24 && lon >= 77 && lon < 81) return 'Madhya Pradesh';
  if (lat >= 29 && lat < 32 && lon >= 74 && lon < 77) return 'Punjab plains';
  if (lat >= 26 && lat < 28 && lon >= 93 && lon < 96) return 'Assam / Northeast';
  if (lat >= 11 && lat < 14 && lon >= 92 && lon < 95) return 'Andaman Sea region';
  if (lat >= 8 && lat < 13 && lon >= 76 && lon < 81) return 'Tamil Nadu / Kerala coast';
  if (lat >= 15 && lat < 20 && lon >= 74 && lon < 81) return 'Maharashtra / Karnataka plateau';
  if (lat >= 22 && lat < 28 && lon >= 75 && lon < 80) return 'Madhya Pradesh / Uttar Pradesh';
  if (lat >= 23 && lat < 28 && lon >= 83 && lon < 88) return 'Chhattisgarh / Jharkhand belt';
  return 'India';
}

function fmtDist(v: number | null): string {
  return v === null ? 'no mapped site within 15 km' : `${v.toLocaleString()} m away`;
}

export default function HumanAnalystView({ hotspot }: Props) {
  const c = hotspot.classification;
  const s = hotspot.spatial_context;
  const place = inferLocation(hotspot.lat, hotspot.lon);

  return (
    <div>
      <h3 className="flex items-center gap-2 mb-3 text-xs font-semibold uppercase tracking-wider text-slate-500 dark:text-slate-400">
        <svg className="w-3 h-3 text-slate-400" fill="none" stroke="currentColor" viewBox="0 0 24 24" aria-hidden="true">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M16 7a4 4 0 11-8 0 4 4 0 018 0zM12 14a7 7 0 00-7 7h14a7 7 0 00-7-7z" />
        </svg>
        What would a human analyst see?
      </h3>
      <div className="text-sm text-slate-700 dark:text-slate-300 space-y-2 leading-relaxed">
        <p>
          A <strong className="font-mono font-medium">{hotspot.frp} MW</strong> fire was detected near <strong>{place}</strong> (
          {hotspot.lat.toFixed(3)}, {hotspot.lon.toFixed(3)}) on {hotspot.acq_date} during the{' '}
          {hotspot.daynight === 'N' ? 'night' : 'day'}.
        </p>
        <p>
          Nearest industrial facility: {fmtDist(s.industry)}; nearest forest: {fmtDist(s.forest)};
          nearest agricultural land: {fmtDist(s.agriculture)}.
        </p>
        <p>
          {hotspot.rule_engine.active_votes} expert rule
          {hotspot.rule_engine.active_votes === 1 ? '' : 's'} voted for{' '}
          <strong>{c.display_name}</strong>; the ML model is {(c.confidence * 100).toFixed(0)}%
          confident ({c.confidence_level.toLowerCase()} band).
        </p>
        <p className="font-semibold">
          {c.requires_human_review
            ? `Recommendation: flagged for human analyst review — ${c.review_reason ?? 'low confidence.'}`
            : 'Recommendation: auto-classified — no human review required.'}
        </p>
      </div>
    </div>
  );
}
