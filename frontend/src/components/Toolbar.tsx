import { ClassKey, Hotspot } from '../types/hotspot';
import { ConfFilter, ReviewFilter } from '../hooks/useAppState';
import PlaceSearch from './PlaceSearch';
import FilterDropdown from './FilterDropdown';
import ViewOptionsDropdown from './ViewOptionsDropdown';

interface ToolbarProps {
  hotspots: Hotspot[];
  activeClasses: ClassKey[];
  confFilter: ConfFilter;
  reviewFilter: ReviewFilter;
  onlyClassified: boolean;
  source: string;
  isLoading: boolean;
  toggleClass: (c: ClassKey) => void;
  selectAllClasses: (on: boolean) => void;
  setConfFilter: (c: ConfFilter) => void;
  setReviewFilter: (r: ReviewFilter) => void;
  toggleOnlyClassified: () => void;
  onDemoSelect: (i: number) => void;
  onFlyToCoords: (lat: number, lon: number) => void;
  onSelectNearest: (lat: number, lon: number) => void;
  onPlacePin: (lat: number, lon: number, label: string) => void;
  onClearPins: () => void;
  loadFromApi: () => void;
  showToast: (msg: string, type?: any) => void;
}

export default function Toolbar(props: ToolbarProps) {
  const {
    hotspots,
    activeClasses,
    confFilter,
    reviewFilter,
    onlyClassified,
    source,
    isLoading,
    toggleClass,
    selectAllClasses,
    setConfFilter,
    setReviewFilter,
    toggleOnlyClassified,
    onDemoSelect,
    onFlyToCoords,
    onSelectNearest,
    onPlacePin,
    onClearPins,
    loadFromApi,
    showToast,
  } = props;

  return (
    <div>
      <div className="flex flex-col gap-4 overflow-visible">
        {/* FILTERS — class + view dropdowns */}
        <div className="min-w-[200px]">
          <div className="text-xs font-medium uppercase tracking-wider text-slate-400 dark:text-slate-500 mb-2">Filters</div>
          <div className="flex flex-wrap items-center gap-2">
            <FilterDropdown
              activeClasses={activeClasses}
              toggleClass={toggleClass}
              selectAllClasses={selectAllClasses}
              panelAlign="right"
            />
            <ViewOptionsDropdown
              confFilter={confFilter}
              reviewFilter={reviewFilter}
              onlyClassified={onlyClassified}
              setConfFilter={setConfFilter}
              setReviewFilter={setReviewFilter}
              toggleOnlyClassified={toggleOnlyClassified}
              panelAlign="right"
            />
          </div>
        </div>

        {/* SEARCH */}
        <div className="min-w-[220px] flex-1">
          <div className="text-xs font-medium uppercase tracking-wider text-slate-400 dark:text-slate-500 mb-2">Search</div>
          <PlaceSearch
            hotspots={hotspots}
            onFlyToCoords={onFlyToCoords}
            onSelectNearest={onSelectNearest}
            onPlacePin={onPlacePin}
            onClearPins={onClearPins}
            showToast={showToast}
          />
        </div>

        {/* ACTIONS */}
        <div>
          <div className="text-xs font-medium uppercase tracking-wider text-slate-400 dark:text-slate-500 mb-2">Actions</div>
          <div className="flex gap-2">
            <button
              onClick={loadFromApi}
              disabled={isLoading}
              className="inline-flex items-center gap-1.5 bg-sky-600 hover:bg-sky-700 active:bg-sky-800 text-white text-xs font-semibold px-4 py-2 rounded-lg shadow-sm transition-all duration-150 ease-in-out hover:scale-[1.02] active:scale-[0.98] disabled:opacity-50 disabled:cursor-not-allowed focus-visible:ring-2 focus-visible:ring-sky-400"
            >
              {isLoading ? <span className="w-3.5 h-3.5 border-2 border-white/40 border-t-white rounded-full animate-spin" /> : (
                <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15" /></svg>
              )}
              {isLoading ? 'Loading...' : 'Run Live Analysis'}
            </button>
            {source === 'demo' && (
              <select
                defaultValue=""
                onChange={(e) => {
                  if (e.target.value !== '') onDemoSelect(Number(e.target.value));
                  e.target.value = '';
                }}
                className="bg-slate-50 dark:bg-slate-700 border border-slate-200 dark:border-slate-700 rounded-lg px-3 py-2 text-xs font-medium text-slate-700 dark:text-slate-200 focus:ring-2 focus:ring-sky-400 focus:outline-none"
              >
                <option value="" disabled>Jump to Demo…</option>
                <option value="0">⭐ Gujarat Industrial Fire</option>
                <option value="1">🏭 Maharashtra Industrial</option>
                <option value="2">🔥 Gujarat Gas Flare</option>
                <option value="3">🌾 Punjab Stubble Burn</option>
                <option value="4">🌲 MP Forest Fire</option>
                <option value="5">⛽ Assam Gas Flare</option>
                <option value="6">🌾 Haryana Burn</option>
                <option value="7">🌲 Uttarakhand Forest</option>
                <option value="8">❓ Unclassified Remote</option>
                <option value="9">⚠️ Borderline Case</option>
              </select>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}
