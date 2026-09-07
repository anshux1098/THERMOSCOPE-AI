import { useCallback, useEffect, useMemo, useState } from 'react';
import { PageKey, useAppState } from './hooks/useAppState';
import NavBar from './components/NavBar';
import Dashboard from './components/Dashboard';
import Analytics from './components/Analytics';
import DataTable from './components/DataTable';
import Methodology from './components/Methodology';
import Alerts from './components/Alerts';
import Toast from './components/Toast';
import SplashScreen from './components/SplashScreen';
import { alertCounts, buildAlerts } from './services/alerts';

export default function App() {
  const state = useAppState();
  const {
    hotspots,
    activePage,
    setActivePage,
    setSelectedId,
    toastMessage,
    toastVisible,
    toastType,
    setSelectedIds,
    showToast,
  } = state;

  const [splashVisible, setSplashVisible] = useState(true);
  useEffect(() => {
    const t = window.setTimeout(() => setSplashVisible(false), 800);
    return () => window.clearTimeout(t);
  }, []);

  const [pendingFlyTo, setPendingFlyTo] = useState<number | null>(null);

  /** From DataTable / Analytics: open the hotspot on the Dashboard map. */
  const handleViewOnMap = useCallback(
    (index: number) => {
      setSelectedId(index);
      setPendingFlyTo(index);
      setActivePage('dashboard');
    },
    [setSelectedId, setActivePage],
  );

  const toggleSelectAll = useCallback(
    (indices: number[]) => {
      setSelectedIds((prev) => {
        const next = new Set(prev);
        const allIn = indices.every((i) => next.has(i));
        if (allIn) indices.forEach((i) => next.delete(i));
        else indices.forEach((i) => next.add(i));
        return next;
      });
    },
    [setSelectedIds],
  );

  const clearSelection = useCallback(() => setSelectedIds(new Set()), [setSelectedIds]);

  const liveAlertCount = useMemo(() => {
    const live = buildAlerts(state.hotspots).filter((a) => {
      const h = state.hotspots[a.hotspotIndex];
      return h && !state.dismissedIds.has(h.id);
    });
    return alertCounts(live).critical;
  }, [state.hotspots, state.dismissedIds]);

  // ESC clears panel selection
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.key === 'Escape') {
        setSelectedId(null);
      }
    };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, [setSelectedId]);

  const page: PageKey = activePage;

  return (
    <div className="h-full flex flex-col bg-[#f8fafc] dark:bg-slate-900 text-slate-900 dark:text-slate-100">
      <SplashScreen visible={splashVisible} />
      <NavBar activePage={page} setActivePage={setActivePage} alertCount={liveAlertCount} />

      <div key={page} className="flex-1 flex flex-col min-h-0 page-enter">
        {page === 'dashboard' && (
          <Dashboard
            appState={state}
            pendingFlyTo={pendingFlyTo}
            clearPendingFlyTo={() => setPendingFlyTo(null)}
          />
        )}
        {page === 'analytics' && (
          <Analytics
            hotspots={hotspots}
            isLoading={state.isLoading}
            selectedIds={state.selectedIds}
            toggleSelectAllVisible={toggleSelectAll}
            clearSelection={clearSelection}
            onViewOnMap={handleViewOnMap}
            showToast={showToast}
          />
        )}
        {page === 'data-table' && (
          <DataTable hotspots={hotspots} onViewOnMap={handleViewOnMap} showToast={showToast} />
        )}
        {page === 'methodology' && <Methodology />}
        {page === 'alerts' && (
          <Alerts
            hotspots={hotspots}
            dismissedIds={state.dismissedIds}
            onDismiss={state.dismissAlert}
            onViewOnMap={handleViewOnMap}
          />
        )}
      </div>

      <Toast message={toastMessage} visible={toastVisible} type={toastType} />
    </div>
  );
}
