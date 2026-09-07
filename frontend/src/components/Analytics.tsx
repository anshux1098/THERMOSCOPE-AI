import { useMemo, useRef, useState } from 'react';
import { Hotspot } from '../types/hotspot';
import { useDateFilter } from '../hooks/useDateFilter';
import { useHeatmap } from '../hooks/useHeatmap';
import { useClustering } from '../hooks/useClustering';
import { formatDate } from '../services/dateUtils';
import MapView, { MapApi } from './Map';
import TimeFilter from './TimeFilter';
import HeatmapOverlay from './HeatmapOverlay';
import ClusterManager from './ClusterManager';
import BatchAnalysisPanel from './BatchAnalysisPanel';
import ConfusionMatrixView from './sections/ConfusionMatrixView';
import LoadingOverlay from './LoadingOverlay';
import EmptyState from './EmptyState';
import { hotspotsToCSV } from './BulkActionsBar';

interface Props {
  hotspots: Hotspot[];
  isLoading: boolean;
  selectedIds: Set<number>;
  toggleSelectAllVisible: (indices: number[]) => void;
  clearSelection: () => void;
  onViewOnMap: (index: number) => void;
  showToast: (msg: string, type?: 'info' | 'success' | 'warning' | 'error') => void;
}

export default function Analytics(props: Props) {
  const { hotspots, isLoading, selectedIds, toggleSelectAllVisible, clearSelection, onViewOnMap, showToast } = props;

  const {
    startDate,
    endDate,
    setStartDate,
    setEndDate,
    filtered: dateFiltered,
    dateRange,
    clearRange,
    isActive: dateActive,
  } = useDateFilter(hotspots);
  const { heatmapOn, toggleHeatmap, intensity, setIntensity, points: heatPoints } = useHeatmap(dateFiltered);
  const { clusteringOn, toggleClustering } = useClustering();
  const [timeVisible, setTimeVisible] = useState(false);
  const [batchVisible, setBatchVisible] = useState(false);

  const mapApiRef = useRef<MapApi | null>(null);

  const dateIdx = useMemo(() => {
    const idx = new Map<Hotspot, number>();
    hotspots.forEach((h, i) => idx.set(h, i));
    return dateFiltered.map((h) => idx.get(h) as number).filter((i) => i !== undefined);
  }, [dateFiltered, hotspots]);

  const handleExport = () => {
    const rows = dateIdx.map((i) => hotspots[i]).filter(Boolean);
    const csv = hotspotsToCSV(rows);
    const blob = new Blob([csv], { type: 'text/csv' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = 'thermoscope-analytics.csv';
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
    showToast(`Exported ${rows.length} hotspots to CSV`, 'success');
  };

  return (
    <div className="flex-1 flex min-h-0">
      {/* Map with layers */}
      <div className="flex-1 flex relative overflow-hidden min-w-0">
        <MapView
          hotspots={hotspots}
          visibleIndices={dateIdx}
          selectedId={null}
          selectedIds={selectedIds}
          clusteringOn={clusteringOn}
          heatmapOn={heatmapOn}
          heatPoints={heatPoints}
          pins={[]}
          onMarkerClick={onViewOnMap}
          onPlacePin={() => {}}
          onClearPins={() => {}}
          registerMapApi={(api) => {
            mapApiRef.current = api;
          }}
        />
        <LoadingOverlay visible={isLoading} />
        <EmptyState visible={!isLoading && dateIdx.length === 0} onClear={clearRange} />
        {timeVisible && (
          <TimeFilter
            hotspots={hotspots}
            startDate={startDate}
            endDate={endDate}
            minDate={dateRange ? formatDate(dateRange.min) : null}
            maxDate={dateRange ? formatDate(dateRange.max) : null}
            setStartDate={setStartDate}
            setEndDate={setEndDate}
            clearRange={clearRange}
            isActive={dateActive}
            open
            onClose={() => setTimeVisible(false)}
          />
        )}
        <HeatmapOverlay visible={heatmapOn} intensity={intensity} setIntensity={setIntensity} />
        {dateActive && startDate && endDate && (
          <div className="absolute top-2 left-1/2 -translate-x-1/2 z-[13] bg-sky-100 text-sky-800 text-[11px] font-bold px-3 py-1.5 rounded-xl shadow">
            Filtered by date: {startDate} to {endDate} ({dateIdx.length} of {hotspots.length})
          </div>
        )}
      </div>

      {/* Sidebar: controls + charts + batch */}
      <aside className="w-80 max-w-[85vw] shrink-0 overflow-y-auto border-l border-slate-200 dark:border-slate-700 bg-slate-50 dark:bg-slate-900 p-3 space-y-3">
        <div className="bg-white dark:bg-slate-800 border border-slate-200 dark:border-slate-700 rounded-2xl shadow p-3">
          <h3 className="text-xs font-bold text-slate-800 dark:text-slate-100 mb-2">Layers &amp; Panels</h3>
          <div className="flex flex-wrap gap-2">
            <button
              type="button"
              aria-pressed={timeVisible}
              onClick={() => setTimeVisible((v) => !v)}
              className={`text-xs font-semibold px-3 py-1.5 rounded-lg border ${
                timeVisible
                  ? 'bg-sky-100 dark:bg-sky-900 border-sky-400'
                  : 'border-slate-300 dark:border-slate-600 text-slate-700 dark:text-slate-200'
              }`}
            >
              📅 Time
            </button>
            <button
              type="button"
              aria-pressed={batchVisible}
              onClick={() => setBatchVisible((v) => !v)}
              className={`text-xs font-semibold px-3 py-1.5 rounded-lg border ${
                batchVisible
                  ? 'bg-violet-100 dark:bg-violet-900 border-violet-400'
                  : 'border-slate-300 dark:border-slate-600 text-slate-700 dark:text-slate-200'
              }`}
            >
              📊 Batch
            </button>
            <button
              type="button"
              aria-pressed={heatmapOn}
              onClick={toggleHeatmap}
              className={`text-xs font-semibold px-3 py-1.5 rounded-lg border ${
                heatmapOn
                  ? 'bg-orange-100 dark:bg-orange-900 border-orange-400'
                  : 'border-slate-300 dark:border-slate-600 text-slate-700 dark:text-slate-200'
              }`}
            >
              🔥 Heatmap
            </button>
            <ClusterManager clusteringOn={clusteringOn} toggleClustering={toggleClustering} visibleCount={dateIdx.length} />
          </div>
          <p className="text-[10px] text-slate-400 mt-2">
            Click a marker to open it on the Dashboard. Heatmap colors encode FRP intensity.
          </p>
        </div>

        <ConfusionMatrixView hotspots={dateFiltered} open floating={false} onClose={() => {}} />

        <BatchAnalysisPanel
          open={batchVisible}
          onClose={() => setBatchVisible(false)}
          hotspots={hotspots}
          visibleIndices={dateIdx}
          selectedIds={selectedIds}
          toggleSelectAllVisible={() => toggleSelectAllVisible(dateIdx)}
          clearSelection={clearSelection}
          onDownload={handleExport}
          showToast={showToast}
        />
      </aside>
    </div>
  );
}
