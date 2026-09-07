import { useEffect, useRef } from 'react';
import { Hotspot, CLASS_COLORS, CLASS_DISPLAY_NAMES, ClassKey } from '../types/hotspot';
import CollapsibleSection from './CollapsibleSection';

interface Props {
  hotspots: Hotspot[];
  total: number;
  classified: number;
  flagged: number;
  /** Compact layout for narrow floating containers: stacked stats, two-line rows. */
  compact?: boolean;
}

const ORDER: ClassKey[] = [
  'industrial_fire',
  'gas_flare',
  'mining_activity',
  'agricultural_burn',
  'forest_natural_fire',
  'industrial_process_heat',
  'unclassified',
];

function useCountUp(target: number) {
  const ref = useRef<HTMLDivElement>(null);
  useEffect(() => {
    const el = ref.current;
    if (!el) return;
    let raf = 0;
    const start = performance.now();
    const duration = 800;
    const tick = (now: number) => {
      const p = Math.min(1, (now - start) / duration);
      const val = Math.round(target * p);
      el.textContent = val.toLocaleString();
      if (p < 1) raf = requestAnimationFrame(tick);
    };
    raf = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(raf);
  }, [target]);
  return ref;
}

function StatCard({ label, value, color, icon, compact = false }: { label: string; value: number; color: string; icon: string; compact?: boolean }) {
  const ref = useCountUp(value);
  return (
    <div className={`flex-1 min-w-[140px] bg-white dark:bg-slate-800 rounded-xl border border-slate-200 dark:border-slate-700 shadow-[0_2px_8px_-2px_rgba(0,0,0,0.06)] dark:shadow-[0_2px_8px_-2px_rgba(0,0,0,0.3)] text-center ${compact ? 'p-3' : 'p-5'}`}>
      <div className="text-slate-400 dark:text-slate-500 mb-1 flex justify-center" aria-hidden="true">
        <span className="text-lg">{icon}</span>
      </div>
      <div ref={ref} className={`font-bold tracking-tight ${compact ? 'text-xl' : 'text-2xl md:text-3xl'} ${color}`}>0</div>
      <div className="text-xs font-medium uppercase tracking-wider text-slate-500 dark:text-slate-400 mt-1">{label}</div>
    </div>
  );
}

export default function StatsPanel({ hotspots, total, classified, flagged, compact = false }: Props) {
  const dist = (() => {
    const counts = new Map<ClassKey, number>();
    for (const h of hotspots) {
      const c = h.classification.class as ClassKey;
      counts.set(c, (counts.get(c) ?? 0) + 1);
    }
    return ORDER.map((k) => ({ key: k, count: counts.get(k) ?? 0 }))
      .sort((a, b) => b.count - a.count);
  })();

  const summary = `${total.toLocaleString()} analyzed · ${classified} classified · ${flagged} flagged`;

  return (
    <CollapsibleSection title="Overview" summary={summary} defaultOpen={false}>
      {/* Stats row */}
      <div>
        <div className={`flex gap-4 ${compact ? 'flex-col gap-2' : 'flex-col md:flex-row'}`}>
          <StatCard label="Analyzed" value={total} color="text-slate-900 dark:text-slate-50" icon="◉" compact={compact} />
          <StatCard label="Classified" value={classified} color="text-emerald-600 dark:text-emerald-400" icon="✓" compact={compact} />
          <StatCard label="Flagged for Review" value={flagged} color="text-amber-600 dark:text-amber-400" icon="⚑" compact={compact} />
        </div>
        <div className="text-xs text-slate-500 dark:text-slate-400 mt-3 text-center">
          From {total.toLocaleString()} FIRMS thermal detections over India — {classified} auto-classified, {flagged} flagged for human review
        </div>
      </div>

      {/* Distribution */}
      <div className="mt-4">
        <div className="flex items-center justify-between mb-3">
          <h3 className="flex items-center gap-2 text-xs font-semibold uppercase tracking-wider text-slate-500 dark:text-slate-400">
            <svg className="w-3.5 h-3.5 text-slate-400" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z" /></svg>
            Classification Distribution
          </h3>
          <span className="text-xs text-slate-500 dark:text-slate-400">Total: {total.toLocaleString()}</span>
        </div>
        <div className="space-y-2">
          {dist.map(({ key, count }) => {
            const pct = total > 0 ? (count / total) * 100 : 0;
            if (compact) {
              return (
                <div key={key}>
                  <div className="flex items-center gap-2">
                    <span className="w-2.5 h-2.5 rounded-full shrink-0" style={{ backgroundColor: CLASS_COLORS[key] }} />
                    <span className="text-[13px] font-medium text-slate-700 dark:text-slate-300 flex-1">{CLASS_DISPLAY_NAMES[key]}</span>
                    <span className="text-[13px] font-semibold text-slate-900 dark:text-slate-50">{count}</span>
                    <span className="text-[11px] text-slate-400 dark:text-slate-500 w-10 text-right">{pct.toFixed(1)}%</span>
                  </div>
                  <div className="h-1.5 rounded-full bg-slate-100 dark:bg-slate-700 overflow-hidden mt-1 ml-[18px]">
                    <div className="h-full rounded-full transition-all duration-300 ease-out" style={{ width: `${pct}%`, backgroundColor: CLASS_COLORS[key] }} />
                  </div>
                </div>
              );
            }
            return (
              <div key={key} className="flex items-center gap-3">
                <span className="w-3 h-3 rounded-full shrink-0" style={{ backgroundColor: CLASS_COLORS[key] }} />
                <span className="text-sm font-medium text-slate-700 dark:text-slate-300 min-w-[140px]">{CLASS_DISPLAY_NAMES[key]}</span>
                <span className="text-sm font-semibold text-slate-900 dark:text-slate-50 w-10 text-right">{count}</span>
                <div className="flex-1 h-2 rounded-full bg-slate-100 dark:bg-slate-700 overflow-hidden">
                  <div className="h-full rounded-full transition-all duration-300 ease-out" style={{ width: `${pct}%`, backgroundColor: CLASS_COLORS[key] }} />
                </div>
                <span className="text-xs text-slate-400 dark:text-slate-500 w-12 text-right">{pct.toFixed(1)}%</span>
              </div>
            );
          })}
        </div>
      </div>
    </CollapsibleSection>
  );
}
