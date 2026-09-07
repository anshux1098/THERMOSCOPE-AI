import { useMemo } from 'react';
import { Hotspot } from '../types/hotspot';
import { countByDate, formatDate } from '../services/dateUtils';

interface Props {
  hotspots: Hotspot[];
  startDate: string | null;
  endDate: string | null;
  minDate: string | null;
  maxDate: string | null;
  setStartDate: (d: string | null) => void;
  setEndDate: (d: string | null) => void;
  clearRange: () => void;
  isActive: boolean;
  open: boolean;
  onClose: () => void;
}

export default function TimeFilter(props: Props) {
  const { hotspots, startDate, endDate, minDate, maxDate, setStartDate, setEndDate, clearRange, isActive, open, onClose } = props;
  const histogram = useMemo(() => countByDate(hotspots), [hotspots]);
  const maxCount = Math.max(1, ...histogram.map((b) => b.count));

  if (!open) return null;

  return (
    <div className="absolute left-3 top-3 z-[13] w-72 max-w-[85vw] bg-white/95 dark:bg-slate-800/95 backdrop-blur-sm border border-slate-200 dark:border-slate-600 rounded-2xl shadow-xl p-3">
      <div className="flex items-center justify-between mb-2">
        <h3 className="text-xs font-bold text-slate-800 dark:text-slate-100">Time Filter</h3>
        <button
          type="button"
          aria-label="Close time filter"
          className="w-6 h-6 flex items-center justify-center rounded-lg hover:bg-slate-100 dark:hover:bg-slate-700 text-slate-500 leading-none"
          onClick={onClose}
        >
          ×
        </button>
      </div>

      {histogram.length === 0 ? (
        <p className="text-xs text-slate-500 dark:text-slate-400">
          No date information available in current dataset.
        </p>
      ) : (
        <>
          <div className="space-y-1 max-h-40 overflow-y-auto pr-1" role="img" aria-label="Hotspots per date">
            {histogram.map((b) => {
              const inRange =
                (!startDate || b.date >= startDate) && (!endDate || b.date <= endDate);
              return (
                <div key={b.date} className="flex items-center gap-2">
                  <span className="text-[10px] font-mono text-slate-500 dark:text-slate-400 w-[70px] shrink-0">
                    {b.date.slice(5)}
                  </span>
                  <div className="flex-1 h-3 bg-slate-100 dark:bg-slate-700 rounded overflow-hidden">
                    <div
                      className={`h-full rounded ${inRange ? 'bg-red-500' : 'bg-slate-300 dark:bg-slate-600'}`}
                      style={{ width: `${Math.max(3, (b.count / maxCount) * 100)}%` }}
                    />
                  </div>
                  <span className="text-[10px] text-slate-500 dark:text-slate-400 w-8 text-right">{b.count}</span>
                </div>
              );
            })}
          </div>

          <div className="flex items-center gap-2 mt-3">
            <label className="text-[11px] text-slate-500 dark:text-slate-400">
              From
              <input
                type="date"
                className="ml-1 text-[11px] border border-slate-300 dark:border-slate-600 rounded px-1 py-0.5 bg-white dark:bg-slate-800 text-slate-700 dark:text-slate-200"
                value={startDate ?? minDate ?? ''}
                min={minDate ?? undefined}
                max={endDate ?? maxDate ?? undefined}
                onChange={(e) => setStartDate(e.target.value || null)}
              />
            </label>
            <label className="text-[11px] text-slate-500 dark:text-slate-400">
              To
              <input
                type="date"
                className="ml-1 text-[11px] border border-slate-300 dark:border-slate-600 rounded px-1 py-0.5 bg-white dark:bg-slate-800 text-slate-700 dark:text-slate-200"
                value={endDate ?? maxDate ?? ''}
                min={startDate ?? minDate ?? undefined}
                max={maxDate ?? undefined}
                onChange={(e) => setEndDate(e.target.value || null)}
              />
            </label>
          </div>

          <div className="flex items-center justify-between mt-2">
            <span className="text-[10px] text-slate-400">
              {minDate && maxDate ? `${formatDate(new Date(`${minDate}T00:00:00`))} → ${formatDate(new Date(`${maxDate}T00:00:00`))}` : ''}
            </span>
            <button
              type="button"
              disabled={!isActive}
              onClick={clearRange}
              className="text-[11px] font-semibold text-red-600 hover:text-red-700 disabled:opacity-40"
            >
              Clear range
            </button>
          </div>
        </>
      )}
    </div>
  );
}
