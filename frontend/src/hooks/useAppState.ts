import { useCallback, useMemo, useRef, useState } from 'react';
import { HOTSPOTS } from '../data/sampleData';
import { ClassKey, Hotspot } from '../types/hotspot';
import { healthCheck } from '../services/api';
import { parseCSV } from '../services/csvParser';

export type Source = 'demo' | 'api' | 'csv';
export type ToastType = 'info' | 'success' | 'warning' | 'error';
export type ConfFilter = 'All' | 'High' | 'Medium' | 'Low';
export type ReviewFilter = 'all' | 'auto' | 'flagged';
export type PageKey = 'dashboard' | 'analytics' | 'data-table' | 'methodology' | 'alerts';

export const ALL_CLASSES: ClassKey[] = [
  'industrial_fire',
  'gas_flare',
  'mining_activity',
  'agricultural_burn',
  'forest_natural_fire',
  'industrial_process_heat',
  'unclassified',
];

const CSV_URL = '/data/classified_hotspots_v2_enriched.csv';

export function useAppState() {
  // Data + load state
  const [hotspots, setHotspots] = useState<Hotspot[]>(HOTSPOTS);
  const [source, setSource] = useState<Source>('demo');
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [serverOnline, setServerOnline] = useState(false);

  // Existing UI state (Phase 1, unchanged)
  const [selectedId, setSelectedId] = useState<number | null>(null);
  const [filterClass, setFilterClass] = useState<ClassKey | 'all'>('all');
  const [onlyClassified, setOnlyClassified] = useState(false);
  const [showMethodology, setShowMethodology] = useState(false);
  const [toastMessage, setToastMessage] = useState('');
  const [toastVisible, setToastVisible] = useState(false);
  const [toastType, setToastType] = useState<ToastType>('info');
  const toastTimer = useRef<number | null>(null);

  // Phase 2 filters (AND-combined with the Phase 1 ones)
  const [activeClasses, setActiveClasses] = useState<ClassKey[]>(ALL_CLASSES);
  const [confFilter, setConfFilter] = useState<ConfFilter>('All');
  const [reviewFilter, setReviewFilter] = useState<ReviewFilter>('all');

  // Phase 4 navigation
  const [activePage, setActivePage] = useState<PageKey>('dashboard');

  // Phase 3 panels + selection
  const [timePanelOpen, setTimePanelOpen] = useState(false);
  const [batchPanelOpen, setBatchPanelOpen] = useState(false);
  const [lfPanelOpen, setLfPanelOpen] = useState(false);
  const [distPanelOpen, setDistPanelOpen] = useState(false);
  const [selectedIds, setSelectedIds] = useState<Set<number>>(new Set());

  // Alerts: session-only dismissals (reset on reload, stated in UI)
  const [dismissedIds, setDismissedIds] = useState<Set<string>>(new Set());
  const dismissAlert = useCallback((id: string) => {
    setDismissedIds((prev) => {
      const next = new Set(prev);
      next.add(id);
      return next;
    });
  }, []);

  const stats = useMemo(
    () => ({
      total: hotspots.length,
      classified: hotspots.filter((h) => !h.classification.requires_human_review).length,
      flagged: hotspots.filter((h) => h.classification.requires_human_review).length,
    }),
    [hotspots],
  );

  const toggleOnlyClassified = useCallback(() => setOnlyClassified((v) => !v), []);

  const showToast = useCallback((message: string, type: ToastType = 'info') => {
    if (toastTimer.current !== null) window.clearTimeout(toastTimer.current);
    setToastMessage(message);
    setToastType(type);
    setToastVisible(true);
    toastTimer.current = window.setTimeout(() => setToastVisible(false), 2600);
  }, []);

  const setLoading = useCallback((v: boolean) => setIsLoading(v), []);
  const setServerOnlineCb = useCallback((v: boolean) => setServerOnline(v), []);
  const setSourceCb = useCallback((s: Source) => setSource(s), []);
  const setErrorCb = useCallback((e: string | null) => setError(e), []);

  const toggleClass = useCallback((c: ClassKey) => {
    setActiveClasses((prev) => (prev.includes(c) ? prev.filter((x) => x !== c) : [...prev, c]));
  }, []);

  const selectAllClasses = useCallback((on: boolean) => {
    setActiveClasses(on ? ALL_CLASSES : []);
  }, []);

  const clearFilters = useCallback(() => {
    setActiveClasses(ALL_CLASSES);
    setConfFilter('All');
    setReviewFilter('all');
    setFilterClass('all');
    setOnlyClassified(false);
  }, []);

  /** Triggered by "Run Live Analysis". Health-check, then CSV; fallback to demo. */
  const loadFromApi = useCallback(async () => {
    setIsLoading(true);
    setError(null);
    try {
      const online = await healthCheck();
      setServerOnline(online);
      const csvText = await fetch(CSV_URL).then((r) => {
        if (!r.ok) throw new Error(`CSV fetch failed (${r.status})`);
        return r.text();
      });
      const parsed = parseCSV(csvText);
      if (parsed.length === 0) throw new Error('CSV parsed to zero hotspots');
      setHotspots(parsed);
      setSource('csv');
      setSelectedId(null);
      showToast(`Loaded ${parsed.length} hotspots${online ? ' (server online)' : ''}`, 'success');
    } catch (e) {
      setServerOnline(false);
      setHotspots(HOTSPOTS);
      setSource('demo');
      const msg = e instanceof Error ? e.message : 'load failed';
      setError(msg);
      showToast('Server offline — using demo data', 'warning');
    } finally {
      setIsLoading(false);
    }
  }, [showToast]);

  return {
    hotspots,
    source,
    isLoading,
    error,
    serverOnline,
    selectedId,
    filterClass,
    onlyClassified,
    activeClasses,
    confFilter,
    reviewFilter,
    activePage,
    timePanelOpen,
    batchPanelOpen,
    lfPanelOpen,
    distPanelOpen,
    selectedIds,
    dismissedIds,
    showMethodology,
    toastMessage,
    toastVisible,
    toastType,
    stats,
    setSelectedId,
    setFilterClass,
    toggleOnlyClassified,
    toggleClass,
    selectAllClasses,
    setConfFilter,
    setReviewFilter,
    clearFilters,
    setActivePage,
    setTimePanelOpen,
    setBatchPanelOpen,
    setLfPanelOpen,
    setDistPanelOpen,
    setSelectedIds,
    dismissAlert,
    setShowMethodology,
    showToast,
    setLoading,
    setError: setErrorCb,
    setServerOnline: setServerOnlineCb,
    setSource: setSourceCb,
    loadFromApi,
  };
}

export type AppState = ReturnType<typeof useAppState>;
