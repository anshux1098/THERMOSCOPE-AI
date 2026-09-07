import { useMemo } from 'react';
import { Hotspot } from '../../types/hotspot';
import { ALL_LF_NAMES } from '../../types/hotspot';

interface Props {
  hotspots: Hotspot[];
  open: boolean;
  onClose: () => void;
  /** When false, renders as a static sidebar card instead of a floating overlay. */
  floating?: boolean;
}

export default function LFCoverageMap({ hotspots, open, onClose, floating = true }: Props) {
  const rows = useMemo(() => {
    const counts = new Map<string, { votes: number; byClass: Map<string, number> }>();
    for (const name of ALL_LF_NAMES) counts.set(name, { votes: 0, byClass: new Map() });
    let totalActive = 0;
    let withVotes = 0;
    for (const h of hotspots) {
      totalActive += h.rule_engine.active_votes;
      if (h.rule_engine.active_votes > 0) withVotes++;
      for (const a of h.rule_engine.active_lfs) {
        const e = counts.get(a.name);
        if (!e) continue;
        e.votes++;
        e.byClass.set(a.vote, (e.byClass.get(a.vote) ?? 0) + 1);
      }
    }
    const max = Math.max(1, ...[...counts.values()].map((e) => e.votes));
    return {
      list: [...counts.entries()]
        .map(([name, e]) => ({ name, ...e, byClass: [...e.byClass.entries()] }))
        .sort((a, b) => b.votes - a.votes),
      max,
      totalActive,
      withVotes,
    };
  }, [hotspots]);

  if (!open) return null;

  return (
    <div
      className={
        floating
          ? 'absolute left-3 top-3 z-[13] w-80 max-w-[85vw] max-h-[70vh] overflow-y-auto bg-white/95 dark:bg-slate-800/95 backdrop-blur-sm border border-slate-200 dark:border-slate-600 rounded-2xl shadow-xl p-4'
          : 'bg-white dark:bg-slate-800 border border-slate-200 dark:border-slate-700 rounded-2xl shadow p-4'
      }
    >
      <div className="flex items-center justify-between mb-1">
        <h3 className="text-sm font-bold text-slate-900 dark:text-slate-100">Labeling Function Coverage</h3>
        {floating && (
          <button
            type="button"
            aria-label="Close LF coverage"
            className="w-6 h-6 flex items-center justify-center rounded-lg hover:bg-slate-100 dark:hover:bg-slate-700 text-slate-500 leading-none"
            onClick={onClose}
          >
            ×
          </button>
        )}
      </div>
      <p className="text-[11px] text-slate-500 dark:text-slate-400 mb-1">
        Votes per LF across all {hotspots.length} hotspots. Low-coverage LFs may need more training data.
      </p>
      <p className="text-[11px] text-slate-500 dark:text-slate-400 mb-3">
        {rows.totalActive} active votes on {rows.withVotes} hotspots.
      </p>
      <div className="space-y-2">
        {rows.list.map((r) => (
          <div key={r.name}>
            <div className="flex justify-between text-[11px]">
              <span className="font-mono text-slate-700 dark:text-slate-200 truncate">{r.name}</span>
              <span className="text-slate-500 dark:text-slate-400 ml-2 shrink-0">{r.votes}</span>
            </div>
            <div className="h-2 bg-slate-100 dark:bg-slate-700 rounded-full overflow-hidden mt-0.5">
              <div className="h-full rounded-full bg-emerald-500" style={{ width: `${(r.votes / rows.max) * 100}%` }} />
            </div>
            {r.byClass.length > 0 && (
              <div className="text-[10px] text-slate-400 mt-0.5">
                {r.byClass.map(([c, n]) => `${c}: ${n}`).join(' · ')}
              </div>
            )}
          </div>
        ))}
      </div>
      <p className="text-[10px] text-slate-400 mt-3">
        Note: file-loaded rows carry vote counts without LF names, so per-LF bars may undercount.
      </p>
    </div>
  );
}
