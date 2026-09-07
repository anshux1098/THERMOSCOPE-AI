import { Hotspot } from '../types/hotspot';

interface Props {
  open: boolean;
  onClose: () => void;
  hotspots: Hotspot[];
  visibleIndices: number[];
  selectedIds: Set<number>;
  toggleSelectAllVisible: () => void;
  clearSelection: () => void;
  onDownload: () => void;
  showToast: (msg: string, type?: 'info' | 'success' | 'warning' | 'error') => void;
}

export default function BatchAnalysisPanel(props: Props) {
  const { open, onClose, hotspots, visibleIndices, selectedIds, toggleSelectAllVisible, clearSelection, onDownload, showToast } = props;
  if (!open) return null;

  const allVisibleSelected = visibleIndices.length > 0 && visibleIndices.every((i) => selectedIds.has(i));

  return (
    <div className="absolute right-3 top-3 z-[13] w-80 max-w-[85vw] max-h-[70vh] overflow-y-auto bg-white/95 dark:bg-slate-800/95 backdrop-blur-sm border border-slate-200 dark:border-slate-600 rounded-2xl shadow-xl p-4">
      <div className="flex items-center justify-between mb-2">
        <h3 className="text-sm font-bold text-slate-900 dark:text-slate-100">Batch Analysis (Phase H)</h3>
        <button
          type="button"
          aria-label="Close batch analysis"
          className="w-6 h-6 flex items-center justify-center rounded-lg hover:bg-slate-100 dark:hover:bg-slate-700 text-slate-500 leading-none"
          onClick={onClose}
        >
          ×
        </button>
      </div>

      <p className="text-xs text-slate-600 dark:text-slate-300 leading-relaxed">
        Select hotspots using filters above, then click <strong>Analyze Selected</strong> to send to the
        backend batch endpoint. Results will be shown in a table and available for download.
      </p>

      <label className="flex items-center gap-2 mt-3 text-xs font-medium text-slate-700 dark:text-slate-200 cursor-pointer">
        <input
          type="checkbox"
          className="rounded border-slate-300 accent-red-600"
          checked={allVisibleSelected}
          onChange={toggleSelectAllVisible}
        />
        Select All Visible ({visibleIndices.length})
      </label>

      <div className="text-xs text-slate-500 dark:text-slate-400 mt-1">
        {selectedIds.size} selected · {hotspots.length} total
      </div>

      <div className="flex flex-wrap gap-2 mt-3">
        <button
          type="button"
          disabled
          title="Batch endpoint not available — Phase H future work"
          className="text-xs font-semibold px-3 py-1.5 rounded-lg bg-slate-200 dark:bg-slate-700 text-slate-400 cursor-not-allowed"
          onClick={() => showToast('Batch endpoint not available — Phase H future work', 'warning')}
        >
          Analyze Selected
        </button>
        <button
          type="button"
          onClick={onDownload}
          className="text-xs font-semibold px-3 py-1.5 rounded-lg border border-slate-300 dark:border-slate-600 text-slate-700 dark:text-slate-200 hover:bg-slate-100 dark:hover:bg-slate-700"
        >
          Download Filtered as CSV
        </button>
        <button
          type="button"
          onClick={clearSelection}
          className="text-xs font-semibold px-3 py-1.5 rounded-lg text-red-600 hover:text-red-700"
        >
          Clear Selection
        </button>
      </div>

      <div className="mt-3 border border-dashed border-slate-300 dark:border-slate-600 rounded-xl p-3 text-center">
        <p className="text-xs text-slate-400">No batch results yet</p>
        <p className="text-[10px] text-slate-400 mt-1">
          Phase H will wire Analyze Selected to POST /api/v1/analysis/batch.
        </p>
      </div>
    </div>
  );
}
