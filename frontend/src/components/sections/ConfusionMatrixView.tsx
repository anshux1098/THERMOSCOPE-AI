import { useMemo } from 'react';
import { ClassKey, CLASS_COLORS, CLASS_DISPLAY_NAMES, Hotspot } from '../../types/hotspot';

interface Props {
  hotspots: Hotspot[];
  open: boolean;
  onClose: () => void;
  /** When false, renders as a static sidebar card instead of a floating overlay. */
  floating?: boolean;
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

export default function ConfusionMatrixView({ hotspots, open, onClose, floating = true }: Props) {
  const rows = useMemo(() => {
    const counts = new Map<ClassKey, number>();
    for (const h of hotspots) {
      const c = h.classification.class as ClassKey;
      counts.set(c, (counts.get(c) ?? 0) + 1);
    }
    return { counts, total: hotspots.length };
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
        <h3 className="text-sm font-bold text-slate-900 dark:text-slate-100">Class Distribution</h3>
        {floating && (
          <button
            type="button"
            aria-label="Close class distribution"
            className="w-6 h-6 flex items-center justify-center rounded-lg hover:bg-slate-100 dark:hover:bg-slate-700 text-slate-500 leading-none"
            onClick={onClose}
          >
            ×
          </button>
        )}
      </div>
      <p className="text-[11px] text-slate-500 dark:text-slate-400 mb-1">
        System classifications on the current dataset ({rows.total} hotspots).
      </p>
      <div className="space-y-2 mt-2">
        {ORDER.map((c) => {
          const n = rows.counts.get(c) ?? 0;
          const pct = rows.total > 0 ? (n / rows.total) * 100 : 0;
          return (
            <div key={c}>
              <div className="flex justify-between text-[11px]">
                <span className="font-medium text-slate-700 dark:text-slate-200">{CLASS_DISPLAY_NAMES[c]}</span>
                <span className="text-slate-500 dark:text-slate-400">
                  {n} ({pct.toFixed(1)}%)
                </span>
              </div>
              <div className="h-2.5 bg-slate-100 dark:bg-slate-700 rounded-full overflow-hidden mt-0.5">
                <div
                  className="h-full rounded-full"
                  style={{ width: `${pct}%`, backgroundColor: CLASS_COLORS[c] }}
                />
              </div>
            </div>
          );
        })}
      </div>
      <p className="text-[10px] text-slate-400 mt-3">
        Not a confusion matrix against ground truth (no labeled holdout set available).
      </p>
    </div>
  );
}
