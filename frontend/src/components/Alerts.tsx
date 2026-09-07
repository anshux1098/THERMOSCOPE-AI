import { useMemo, useState } from 'react';
import { CLASS_COLORS, ClassKey, Hotspot } from '../types/hotspot';
import { AlertItem, buildAlerts, Severity, alertCounts } from '../services/alerts';

interface Props {
  hotspots: Hotspot[];
  dismissedIds: Set<string>;
  onDismiss: (id: string) => void;
  onViewOnMap: (index: number) => void;
}

type SevFilter = 'all' | Severity;
type SortKey = 'newest' | 'risk' | 'confidence';

const SEV_ORDER: Record<Severity, number> = { critical: 0, high: 1, medium: 2, low: 3 };

const SEV_STYLE: Record<Severity, string> = {
  critical: 'bg-red-600 text-white',
  high: 'bg-orange-500 text-white',
  medium: 'bg-amber-400 text-slate-900',
  low: 'bg-slate-200 dark:bg-slate-700 text-slate-700 dark:text-slate-200',
};

const FILTERS: { v: SevFilter; label: string }[] = [
  { v: 'all', label: 'All' },
  { v: 'critical', label: 'Critical' },
  { v: 'high', label: 'High' },
  { v: 'medium', label: 'Medium' },
  { v: 'low', label: 'Low' },
];

export default function Alerts({ hotspots, dismissedIds, onDismiss, onViewOnMap }: Props) {
  const [sevFilter, setSevFilter] = useState<SevFilter>('all');
  const [sortKey, setSortKey] = useState<SortKey>('risk');
  const [expanded, setExpanded] = useState<Set<number>>(new Set());

  const alerts = useMemo(() => {
    const all = buildAlerts(hotspots);
    const live = all.filter((a) => !dismissedIds.has(hotspots[a.hotspotIndex]?.id ?? ''));
    const f = sevFilter === 'all' ? live : live.filter((a) => a.severity === sevFilter);
    const sorted = [...f];
    if (sortKey === 'newest') {
      sorted.sort((a, b) => {
        const da = hotspots[a.hotspotIndex]?.acq_date ?? '';
        const db = hotspots[b.hotspotIndex]?.acq_date ?? '';
        return db.localeCompare(da) || b.hotspotIndex - a.hotspotIndex;
      });
    } else if (sortKey === 'confidence') {
      sorted.sort(
        (a, b) => (hotspots[a.hotspotIndex]?.classification.confidence ?? 1) - (hotspots[b.hotspotIndex]?.classification.confidence ?? 1),
      );
    } else {
      sorted.sort((a, b) => {
        const s = SEV_ORDER[a.severity] - SEV_ORDER[b.severity];
        if (s !== 0) return s;
        return (hotspots[a.hotspotIndex]?.classification.confidence ?? 1) - (hotspots[b.hotspotIndex]?.classification.confidence ?? 1);
      });
    }
    return sorted;
  }, [hotspots, dismissedIds, sevFilter, sortKey]);

  const counts = useMemo(() => {
    const live = buildAlerts(hotspots).filter((a) => !dismissedIds.has(hotspots[a.hotspotIndex]?.id ?? ''));
    return alertCounts(live);
  }, [hotspots, dismissedIds]);

  const toggleExpand = (i: number) => {
    setExpanded((prev) => {
      const next = new Set(prev);
      if (next.has(i)) next.delete(i);
      else next.add(i);
      return next;
    });
  };

  const row = (a: AlertItem) => {
    const h = hotspots[a.hotspotIndex];
    if (!h) return null;
    const isOpen = expanded.has(a.hotspotIndex);
    return (
      <div
        key={`${h.id}-${a.hotspotIndex}`}
        className="bg-white dark:bg-slate-800 border border-slate-200 dark:border-slate-700 rounded-xl shadow-[0_2px_8px_-2px_rgba(0,0,0,0.06)] dark:shadow-[0_2px_8px_-2px_rgba(0,0,0,0.3)] p-4"
      >
        <div className="flex flex-wrap items-center gap-2">
          <span className={`text-[10px] font-bold uppercase tracking-wider px-2 py-0.5 rounded-full ${SEV_STYLE[a.severity]}`}>
            {a.severity}
          </span>
          <span className="flex items-center gap-1.5 text-sm font-semibold text-slate-900 dark:text-slate-50">
            <span
              className="rounded-full inline-block"
              style={{ width: 10, height: 10, backgroundColor: CLASS_COLORS[h.classification.class as ClassKey] ?? '#9e9e9e' }}
            />
            {h.classification.display_name}
          </span>
          <span className="font-mono text-xs text-slate-500 dark:text-slate-400">
            {h.lat.toFixed(3)}, {h.lon.toFixed(3)} · {h.frp} MW · {h.acq_date}
          </span>
          <span className="ml-auto flex items-center gap-1.5">
            <button
              type="button"
              onClick={() => onViewOnMap(a.hotspotIndex)}
              className="text-xs font-semibold text-sky-600 dark:text-sky-400 hover:underline rounded px-1 py-1 focus-visible:ring-2 focus-visible:ring-sky-400 focus:outline-none"
            >
              View on Map
            </button>
            <button
              type="button"
              aria-label={`Expand alert ${h.id}`}
              onClick={() => toggleExpand(a.hotspotIndex)}
              className="w-6 h-6 flex items-center justify-center rounded-lg hover:bg-slate-100 dark:hover:bg-slate-700 text-slate-500 text-sm leading-none focus-visible:ring-2 focus-visible:ring-sky-400 focus:outline-none"
            >
              <svg className={`w-3.5 h-3.5 transition-transform duration-200 ${isOpen ? 'rotate-180' : ''}`} fill="none" stroke="currentColor" viewBox="0 0 24 24" aria-hidden="true">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
              </svg>
            </button>
            <button
              type="button"
              aria-label={`Dismiss alert ${h.id}`}
              onClick={() => onDismiss(h.id)}
              className="w-6 h-6 flex items-center justify-center rounded-lg hover:bg-slate-100 dark:hover:bg-slate-700 text-slate-400 hover:text-slate-600 dark:hover:text-slate-200 text-sm leading-none focus-visible:ring-2 focus-visible:ring-sky-400 focus:outline-none"
            >
              ×
            </button>
          </span>
        </div>
        <p className="text-xs text-slate-600 dark:text-slate-300 mt-1.5">{a.reason}</p>
        <p className="text-xs text-slate-700 dark:text-slate-200 mt-1">
          <strong>Recommended:</strong> {a.actions[0]?.action}
        </p>
        {isOpen && a.actions.length > 1 && (
          <ul className="mt-1.5 space-y-1 border-t border-slate-100 dark:border-slate-700 pt-1.5">
            {a.actions.slice(1).map((act, i) => (
              <li key={i} className="text-xs text-slate-600 dark:text-slate-300 flex gap-1.5">
                <span className="font-bold text-slate-400 dark:text-slate-500 shrink-0">[{act.priority}]</span>
                <span>{act.action}</span>
              </li>
            ))}
          </ul>
        )}
      </div>
    );
  };

  return (
    <div className="flex-1 overflow-y-auto min-h-0">
      <div className="max-w-[900px] mx-auto p-4 space-y-3">
        <div className="flex flex-wrap items-end gap-x-6 gap-y-2">
          <div>
            <h2 className="font-bold text-slate-900 dark:text-slate-100 text-xl">Alerts</h2>
            <p className="text-xs text-slate-500 dark:text-slate-400">Triage queue — review, then act.</p>
          </div>
          <div className="flex items-center gap-4 ml-auto text-xs">
            {(['critical', 'high', 'medium', 'low'] as Severity[]).map((s) => (
              <span key={s} className="flex items-baseline gap-1">
                <span className={`font-bold font-mono text-base ${s === 'critical' ? 'text-red-600 dark:text-red-400' : s === 'high' ? 'text-orange-500' : s === 'medium' ? 'text-amber-600 dark:text-amber-400' : 'text-slate-500'}`}>
                  {counts[s].toLocaleString()}
                </span>
                <span className="text-slate-500 dark:text-slate-400 font-medium">{s}</span>
              </span>
            ))}
          </div>
        </div>

        <div className="flex flex-wrap items-center gap-2">
          <div className="flex items-center gap-1" role="group" aria-label="Severity filter">
            {FILTERS.map((f) => (
              <button
                key={f.v}
                type="button"
                aria-pressed={sevFilter === f.v}
                onClick={() => setSevFilter(f.v)}
                className={`text-[11px] font-semibold px-2.5 py-1 rounded-full border transition-all duration-150 ease-in-out focus-visible:ring-2 focus-visible:ring-sky-400 focus:outline-none ${
                  sevFilter === f.v
                    ? 'bg-slate-900 dark:bg-slate-100 border-slate-900 dark:border-slate-100 text-white dark:text-slate-900'
                    : 'bg-white dark:bg-slate-800 border-slate-200 dark:border-slate-700 text-slate-600 dark:text-slate-300 hover:bg-slate-100 dark:hover:bg-slate-700'
                }`}
              >
                {f.label}
              </button>
            ))}
          </div>
          <select
            aria-label="Sort alerts"
            value={sortKey}
            onChange={(e) => setSortKey(e.target.value as SortKey)}
            className="text-xs border border-slate-300 dark:border-slate-600 rounded-lg px-2 py-1.5 bg-white dark:bg-slate-800 text-slate-700 dark:text-slate-200 focus:outline-none focus-visible:ring-2 focus-visible:ring-sky-400"
          >
            <option value="risk">Highest risk</option>
            <option value="newest">Newest</option>
            <option value="confidence">Lowest confidence</option>
          </select>
        </div>

        {alerts.length === 0 ? (
          <div className="bg-white dark:bg-slate-800 border border-slate-200 dark:border-slate-700 rounded-xl p-8 text-center">
            <p className="text-sm text-slate-500 dark:text-slate-400">No alerts at this severity — queue is clear.</p>
          </div>
        ) : (
          alerts.map(row)
        )}

        <p className="text-[11px] text-slate-400 dark:text-slate-500">
          Severity derives from the same weak-supervision pipeline — treat as triage, not ground truth.
          Dismissed alerts return on reload (session-only).
        </p>
      </div>
    </div>
  );
}
