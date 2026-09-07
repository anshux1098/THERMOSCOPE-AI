import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { ALL_CLASSES, AppState, ConfFilter, ReviewFilter } from '../hooks/useAppState';
import { useKeyboardNav } from '../hooks/useKeyboardNav';
import { ClassKey, Hotspot } from '../types/hotspot';
import Header from './Header';
import Toolbar from './Toolbar';
import StatsPanel from './StatsPanel';
import Map, { MapApi, PlacePin } from './Map';
import DetailPanel from './DetailPanel';
import LegendBar from './Legend';
import LoadingOverlay from './LoadingOverlay';
import EmptyState from './EmptyState';
import ErrorToast from './ErrorToast';
import BulkActionsBar, { hotspotsToCSV } from './BulkActionsBar';

interface Props {
  appState: AppState;
  pendingFlyTo: number | null;
  clearPendingFlyTo: () => void;
}

function passesFilters(
  h: Hotspot,
  filterClass: ClassKey | 'all',
  onlyClassified: boolean,
  activeClasses: ClassKey[],
  confFilter: ConfFilter,
  reviewFilter: ReviewFilter,
): boolean {
  if (filterClass !== 'all' && h.classification.class !== filterClass) return false;
  if (onlyClassified && h.classification.requires_human_review) return false;
  if (!activeClasses.includes(h.classification.class as ClassKey)) return false;
  if (confFilter !== 'All' && h.classification.confidence_level !== confFilter) return false;
  if (reviewFilter === 'auto' && h.classification.requires_human_review) return false;
  if (reviewFilter === 'flagged' && !h.classification.requires_human_review) return false;
  return true;
}

export default function Dashboard({ appState, pendingFlyTo, clearPendingFlyTo }: Props) {
  const {
    hotspots,
    source,
    isLoading,
    error,
    selectedId,
    filterClass,
    onlyClassified,
    activeClasses,
    confFilter,
    reviewFilter,
    stats,
    setSelectedId,
    setFilterClass,
    toggleOnlyClassified,
    toggleClass,
    selectAllClasses,
    setConfFilter,
    setReviewFilter,
    clearFilters,
    setError,
    loadFromApi,
    showToast,
  } = appState;
  const [pins, setPins] = useState<PlacePin[]>([]);
  const [filterCardOpen, setFilterCardOpen] = useState(true);

  const onPlacePin = useCallback(
    (lat: number, lon: number, label: string) => {
      setPins((prev) => [...prev.slice(-9), { lat, lon, label }]);
    },
    [],
  );
  const onClearPins = useCallback(() => setPins([]), []);

  const visibleIndices = useMemo(
    () =>
      hotspots
        .map((h, i) => ({ h, i }))
        .filter(({ h }) => passesFilters(h, filterClass, onlyClassified, activeClasses, confFilter, reviewFilter))
        .map(({ i }) => i),
    [hotspots, filterClass, onlyClassified, activeClasses, confFilter, reviewFilter],
  );

  const mapApiRef = useRef<MapApi | null>(null);

  useEffect(() => {
    if (pendingFlyTo !== null && mapApiRef.current) {
      mapApiRef.current.flyToIndex(pendingFlyTo);
      clearPendingFlyTo();
    }
  }, [pendingFlyTo, clearPendingFlyTo]);

  const handleDemoSelect = useCallback(
    (index: number) => {
      mapApiRef.current?.flyToIndex(index);
    },
    [],
  );

  const handleFlyToCoords = useCallback((lat: number, lon: number) => {
    mapApiRef.current?.flyToCoords(lat, lon, 12);
  }, []);

  const handleSelectNearest = useCallback(
    (lat: number, lon: number) => {
      let best = -1;
      let bestD = Infinity;
      hotspots.forEach((h, i) => {
        const d = (h.lat - lat) ** 2 + (h.lon - lon) ** 2;
        if (d < bestD) {
          bestD = d;
          best = i;
        }
      });
      if (best >= 0) {
        setSelectedId(best);
        mapApiRef.current?.flyToCoords(hotspots[best].lat, hotspots[best].lon, 11);
      }
    },
    [hotspots, setSelectedId],
  );

  const getCoords = useCallback(
    (index: number) => {
      const h = hotspots[index];
      return h ? { lat: h.lat, lon: h.lon } : null;
    },
    [hotspots],
  );

  useKeyboardNav({
    visibleIndices,
    selectedId,
    onSelect: setSelectedId,
    onFlyTo: handleFlyToCoords,
    getCoords,
    toggleClassifiedOnly: toggleOnlyClassified,
  });

  const visibleHotspots = useMemo(
    () => visibleIndices.map((i) => hotspots[i]).filter(Boolean),
    [visibleIndices, hotspots],
  );

  const handleExport = useCallback(() => {
    const csv = hotspotsToCSV(visibleHotspots);
    const blob = new Blob([csv], { type: 'text/csv' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = 'thermoscope-hotspots.csv';
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
    showToast(`Exported ${visibleHotspots.length} hotspots to CSV`, 'success');
  }, [visibleHotspots, showToast]);

  const filtersActive =
    filterClass !== 'all' ||
    onlyClassified ||
    activeClasses.length !== ALL_CLASSES.length ||
    confFilter !== 'All' ||
    reviewFilter !== 'all';

  const filtersDefault =
    filterClass === 'all' &&
    activeClasses.length === ALL_CLASSES.length &&
    confFilter === 'All' &&
    reviewFilter === 'all' &&
    !onlyClassified;

  return (
    <div className="flex-1 flex flex-col min-h-0">
      <Header />

      <div className="flex-1 flex relative overflow-hidden min-h-[400px]">
        <div className="absolute left-3 top-3 z-[14] w-[280px] max-sm:left-3 max-sm:right-3 max-sm:w-auto max-h-[calc(100%-24px)] overflow-y-auto rounded-xl">
          <StatsPanel hotspots={hotspots} total={stats.total} classified={stats.classified} flagged={stats.flagged} compact />
        </div>
        {filterCardOpen ? (
          <div className="absolute right-3 top-3 z-[14] w-[270px] max-sm:left-3 max-sm:right-3 max-sm:w-auto bg-white dark:bg-slate-800 rounded-xl border border-slate-200 dark:border-slate-700 shadow-[0_2px_8px_-2px_rgba(0,0,0,0.06)] dark:shadow-[0_2px_8px_-2px_rgba(0,0,0,0.3)]">
            <div className="flex items-center justify-between px-3 pt-2.5">
              <span className="text-xs font-semibold uppercase tracking-wider text-slate-500 dark:text-slate-400">
                Filters, Search & Actions
              </span>
              <button
                type="button"
                aria-label="Minimize filter panel"
                onClick={() => setFilterCardOpen(false)}
                className="w-6 h-6 flex items-center justify-center rounded-lg hover:bg-slate-100 dark:hover:bg-slate-700 text-slate-500 text-sm leading-none focus-visible:ring-2 focus-visible:ring-offset-2 focus-visible:ring-sky-400 focus:outline-none"
              >
                ×
              </button>
            </div>
            <div className="p-3 pt-1 overflow-visible">
              <Toolbar
                hotspots={hotspots}
                activeClasses={activeClasses}
                confFilter={confFilter}
                reviewFilter={reviewFilter}
                onlyClassified={onlyClassified}
                source={source}
                isLoading={isLoading}
                toggleClass={toggleClass}
                selectAllClasses={selectAllClasses}
                setConfFilter={setConfFilter}
                setReviewFilter={setReviewFilter}
                toggleOnlyClassified={toggleOnlyClassified}
                onDemoSelect={handleDemoSelect}
                onFlyToCoords={handleFlyToCoords}
                onSelectNearest={handleSelectNearest}
                onPlacePin={onPlacePin}
                onClearPins={onClearPins}
                loadFromApi={loadFromApi}
                showToast={showToast}
              />
            </div>
          </div>
        ) : (
          <button
            type="button"
            onClick={() => setFilterCardOpen(true)}
            className="absolute right-3 top-3 z-[14] inline-flex items-center gap-1.5 px-3 py-2 text-xs font-semibold rounded-full border bg-white dark:bg-slate-800 border-slate-200 dark:border-slate-700 text-slate-700 dark:text-slate-200 shadow-[0_2px_8px_-2px_rgba(0,0,0,0.06)] dark:shadow-[0_2px_8px_-2px_rgba(0,0,0,0.3)] hover:bg-slate-50 dark:hover:bg-slate-700 focus-visible:ring-2 focus-visible:ring-offset-2 focus-visible:ring-sky-400 focus:outline-none"
          >
            <svg className="w-3.5 h-3.5 text-slate-400" fill="none" stroke="currentColor" viewBox="0 0 24 24" aria-hidden="true">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M3 4a1 1 0 011-1h16a1 1 0 011 1v2.586a1 1 0 01-.293.707l-6.414 6.414a1 1 0 00-.293.707V17l-4 4v-6.586a1 1 0 00-.293-.707L3.293 7.293A1 1 0 013 6.586V4z" />
            </svg>
            Filters
            {!filtersDefault && <span className="w-2 h-2 rounded-full bg-amber-500" aria-label="filters active" />}
          </button>
        )}
        <Map
          hotspots={hotspots}
          visibleIndices={visibleIndices}
          selectedId={selectedId}
          selectedIds={new Set()}
          clusteringOn={false}
          heatmapOn={false}
          heatPoints={[]}
          pins={pins}
          onMarkerClick={setSelectedId}
          onPlacePin={onPlacePin}
          onClearPins={onClearPins}
          registerMapApi={(api) => {
            mapApiRef.current = api;
          }}
        />
        <LoadingOverlay visible={isLoading} />
        <EmptyState visible={!isLoading && visibleIndices.length === 0} onClear={clearFilters} />
        <ErrorToast message={error} onDismiss={() => setError(null)} />
        {source === 'demo' && !isLoading && (
          <div className="absolute top-2 left-1/2 -translate-x-1/2 z-[13] bg-white/95 dark:bg-slate-800/95 border border-slate-200 dark:border-slate-600 rounded-xl shadow px-3 py-1.5 text-[11px] text-slate-600 dark:text-slate-300 font-medium">
            Demo data (10 hotspots) — click <strong>Run Live Analysis</strong> to load the full dataset
          </div>
        )}
        <BulkActionsBar
          visible={filtersActive && visibleIndices.length > 0}
          shown={visibleIndices.length}
          total={hotspots.length}
          onZoomToFit={() => mapApiRef.current?.fitBoundsVisible()}
          onExport={handleExport}
        />
        <div className="absolute bottom-2 left-1/2 -translate-x-1/2 z-[13] max-w-[95vw]">
          <LegendBar filterClass={filterClass} setFilterClass={setFilterClass} hotspots={hotspots} />
        </div>
        <DetailPanel
          hotspots={hotspots}
          selectedId={selectedId}
          onClose={() => setSelectedId(null)}
          onCopy={(text) => showToast(text, 'success')}
        />
      </div>
    </div>
  );
}
