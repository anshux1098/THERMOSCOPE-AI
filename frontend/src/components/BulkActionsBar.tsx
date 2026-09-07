import { Hotspot } from '../types/hotspot';

interface Props {
  visible: boolean;
  shown: number;
  total: number;
  onZoomToFit: () => void;
  onExport: () => void;
}

export function hotspotsToCSV(hotspots: Hotspot[]): string {
  const header = 'id,lat,lon,frp,bright_ti4,confidence,daynight,acq_date,satellite,class,confidence_pct,decision_source,requires_review';
  const rows = hotspots.map((h) =>
    [
      h.id,
      h.lat,
      h.lon,
      h.frp,
      h.bright_ti4,
      h.confidence,
      h.daynight,
      h.acq_date,
      h.satellite,
      h.classification.class,
      (h.classification.confidence * 100).toFixed(1),
      h.classification.decision_source,
      h.classification.requires_human_review,
    ].join(','),
  );
  return [header, ...rows].join('\n');
}

export default function BulkActionsBar({ visible, shown, total, onZoomToFit, onExport }: Props) {
  if (!visible) return null;
  return (
    <div className="absolute bottom-3 left-3 z-[13] flex items-center gap-2 bg-white/95 dark:bg-slate-800/95 backdrop-blur-sm border border-slate-200 dark:border-slate-600 rounded-xl shadow-[0_2px_8px_-2px_rgba(0,0,0,0.06)] dark:shadow-[0_2px_8px_-2px_rgba(0,0,0,0.3)] px-3 py-2 text-xs">
      <span className="text-slate-600 dark:text-slate-300 font-medium">
        Showing <span className="font-mono font-bold text-slate-800 dark:text-slate-100">{shown}</span> of{' '}
        <span className="font-mono font-bold text-slate-800 dark:text-slate-100">{total}</span> hotspots
      </span>
      <button
        type="button"
        className="font-semibold text-slate-700 dark:text-slate-200 border border-slate-300 dark:border-slate-600 rounded-lg px-2 py-1 hover:bg-slate-100 dark:hover:bg-slate-700 transition-all duration-150 ease-in-out hover:scale-[1.02] active:scale-[0.98] focus-visible:ring-2 focus-visible:ring-offset-2 focus-visible:ring-sky-400"
        onClick={onZoomToFit}
      >
        Zoom to fit
      </button>
      <button
        type="button"
        className="font-semibold text-slate-700 dark:text-slate-200 border border-slate-300 dark:border-slate-600 rounded-lg px-2 py-1 hover:bg-slate-100 dark:hover:bg-slate-700 transition-all duration-150 ease-in-out hover:scale-[1.02] active:scale-[0.98] focus-visible:ring-2 focus-visible:ring-offset-2 focus-visible:ring-sky-400"
        onClick={onExport}
      >
        Export CSV
      </button>
    </div>
  );
}
