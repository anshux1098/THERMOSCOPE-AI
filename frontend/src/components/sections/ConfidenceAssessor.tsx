import { Hotspot } from '../../types/hotspot';

interface Props {
  hotspot: Hotspot;
}

function verdict(frp: number, bright: number, conf: string, daynight: string) {
  const rows: { label: string; detail: string; strength: 'strong' | 'mid' | 'weak' }[] = [];
  let strong = 0;

  if (frp > 15) {
    rows.push({ label: 'FRP', detail: `${frp} MW — High (threshold: > 15 MW)`, strength: 'strong' });
    strong++;
  } else if (frp >= 5) {
    rows.push({ label: 'FRP', detail: `${frp} MW — Moderate (threshold: 5–15 MW)`, strength: 'mid' });
  } else {
    rows.push({ label: 'FRP', detail: `${frp} MW — Low (threshold: < 5 MW)`, strength: 'weak' });
  }

  if (bright > 330) {
    rows.push({ label: 'Brightness', detail: `${bright} K — Hot (threshold: > 330 K)`, strength: 'strong' });
    strong++;
  } else if (bright >= 310) {
    rows.push({ label: 'Brightness', detail: `${bright} K — Warm (threshold: 310–330 K)`, strength: 'mid' });
  } else {
    rows.push({ label: 'Brightness', detail: `${bright} K — Cool (threshold: < 310 K)`, strength: 'weak' });
  }

  if (conf === 'high') {
    rows.push({ label: 'Detection confidence', detail: 'high (FIRMS quality flag) — Strong', strength: 'strong' });
    strong++;
  } else if (conf === 'nominal') {
    rows.push({ label: 'Detection confidence', detail: 'nominal (FIRMS quality flag) — Moderate', strength: 'mid' });
  } else {
    rows.push({ label: 'Detection confidence', detail: `${conf} (FIRMS quality flag) — Weak`, strength: 'weak' });
  }

  rows.push(
    daynight === 'D'
      ? { label: 'Day/Night', detail: 'Day — visible spectrum detection', strength: 'mid' }
      : { label: 'Day/Night', detail: 'Night — thermal-only (infrared)', strength: 'mid' },
  );

  return { rows, strong };
}

const MARK = {
  strong: { icon: '✓', cls: 'text-emerald-600' },
  mid: { icon: '–', cls: 'text-amber-600' },
  weak: { icon: '✕', cls: 'text-red-500' },
} as const;

export default function ConfidenceAssessor({ hotspot }: Props) {
  const { rows, strong } = verdict(hotspot.frp, hotspot.bright_ti4, hotspot.confidence, hotspot.daynight);

  return (
    <div>
      <h3 className="flex items-center gap-2 mb-3 text-xs font-semibold uppercase tracking-wider text-slate-500 dark:text-slate-400">
        <svg className="w-3 h-3 text-slate-400" fill="none" stroke="currentColor" viewBox="0 0 24 24" aria-hidden="true">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z" />
        </svg>
        Is this a real thermal event?
      </h3>
      <div>
        {rows.map((r) => (
          <div key={r.label} className="flex items-start gap-2 py-2 border-b border-slate-100 dark:border-slate-700 last:border-0 text-xs">
            <span className={`font-bold w-4 text-center ${MARK[r.strength].cls}`} aria-hidden="true">
              {MARK[r.strength].icon}
            </span>
            <span className="text-slate-500 dark:text-slate-400 font-medium">{r.label}:</span>
            <span className="font-mono text-sm font-medium text-slate-800 dark:text-slate-200">{r.detail}</span>
          </div>
        ))}
        <div className="mt-2 pt-2 text-xs font-bold text-slate-800 dark:text-slate-100">
          {strong >= 3 ? (
            <span className="text-emerald-700 dark:text-emerald-400">Verdict: ✓ Strong thermal signal ({strong} of 4 indicators strong)</span>
          ) : (
            <span className="text-amber-700 dark:text-amber-400">Verdict: Weak thermal signal — review recommended</span>
          )}
        </div>
      </div>
    </div>
  );
}
