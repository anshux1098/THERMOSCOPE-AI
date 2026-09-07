import { useMemo, useState } from 'react';
import { ClassKey, CLASS_DISPLAY_NAMES, Hotspot } from '../types/hotspot';
import { ALL_CLASSES } from '../hooks/useAppState';
import { hotspotsToCSV } from './BulkActionsBar';

interface Props {
  hotspots: Hotspot[];
  onViewOnMap: (index: number) => void;
  showToast: (msg: string, type?: 'info' | 'success' | 'warning' | 'error') => void;
}

type SortKey = 'id' | 'lat' | 'lon' | 'frp' | 'bright_ti4' | 'confidence' | 'daynight' | 'acq_date' | 'class' | 'decision' | 'score' | 'review';

const COLUMNS: { key: SortKey; label: string }[] = [
  { key: 'id', label: 'ID' },
  { key: 'lat', label: 'Lat' },
  { key: 'lon', label: 'Lon' },
  { key: 'frp', label: 'FRP' },
  { key: 'bright_ti4', label: 'Brightness' },
  { key: 'confidence', label: 'Confidence' },
  { key: 'daynight', label: 'D/N' },
  { key: 'acq_date', label: 'Date' },
  { key: 'class', label: 'Class' },
  { key: 'decision', label: 'Decision' },
  { key: 'score', label: 'Score' },
  { key: 'review', label: 'Review' },
];

function cellValue(h: Hotspot, key: SortKey): string | number {
  switch (key) {
    case 'id':
      return h.id;
    case 'lat':
      return h.lat;
    case 'lon':
      return h.lon;
    case 'frp':
      return h.frp;
    case 'bright_ti4':
      return h.bright_ti4;
    case 'confidence':
      return h.confidence;
    case 'daynight':
      return h.daynight;
    case 'acq_date':
      return h.acq_date;
    case 'class':
      return h.classification.class;
    case 'decision':
      return h.classification.decision_source;
    case 'score':
      return h.classification.confidence;
    case 'review':
      return h.classification.requires_human_review ? 1 : 0;
  }
}

export default function DataTable({ hotspots, onViewOnMap, showToast }: Props) {
  const [query, setQuery] = useState('');
  const [classFilter, setClassFilter] = useState<ClassKey | 'all'>('all');
  const [reviewFilter, setReviewFilter] = useState<'all' | 'auto' | 'flagged'>('all');
  const [sortKey, setSortKey] = useState<SortKey>('frp');
  const [sortDir, setSortDir] = useState<1 | -1>(-1);
  const [perPage, setPerPage] = useState(25);
  const [page, setPage] = useState(0);

  const filtered = useMemo(() => {
    const q = query.trim().toLowerCase();
    return hotspots.filter((h) => {
      if (classFilter !== 'all' && h.classification.class !== classFilter) return false;
      if (reviewFilter === 'auto' && h.classification.requires_human_review) return false;
      if (reviewFilter === 'flagged' && !h.classification.requires_human_review) return false;
      if (!q) return true;
      const hay = [
        h.id,
        String(h.lat),
        String(h.lon),
        String(h.frp),
        String(h.bright_ti4),
        h.confidence,
        h.daynight,
        h.acq_date,
        h.satellite,
        h.classification.class,
        h.classification.display_name,
        h.classification.decision_source,
      ]
        .join(' ')
        .toLowerCase();
      return hay.includes(q);
    });
  }, [hotspots, query, classFilter, reviewFilter]);

  const sorted = useMemo(() => {
    const arr = [...filtered];
    arr.sort((a, b) => {
      const va = cellValue(a, sortKey);
      const vb = cellValue(b, sortKey);
      if (typeof va === 'number' && typeof vb === 'number') return (va - vb) * sortDir;
      return String(va).localeCompare(String(vb)) * sortDir;
    });
    return arr;
  }, [filtered, sortKey, sortDir]);

  const pageCount = Math.max(1, Math.ceil(sorted.length / perPage));
  const safePage = Math.min(page, pageCount - 1);
  const pageRows = sorted.slice(safePage * perPage, safePage * perPage + perPage);

  const clickSort = (key: SortKey) => {
    if (key === sortKey) setSortDir((d) => (d === 1 ? -1 : 1));
    else {
      setSortKey(key);
      setSortDir(-1);
    }
    setPage(0);
  };

  const handleExport = () => {
    const csv = hotspotsToCSV(filtered);
    const blob = new Blob([csv], { type: 'text/csv' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = 'thermoscope-datatable.csv';
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
    showToast(`Exported ${filtered.length} hotspots to CSV`, 'success');
  };

  return (
    <div className="flex-1 overflow-y-auto min-h-0">
      <div className="max-w-[1400px] mx-auto p-4">
        <div className="flex flex-wrap items-center gap-2 mb-3">
          <h2 className="font-bold text-slate-900 dark:text-slate-100 text-lg mr-auto">
            Data Table <span className="text-xs font-medium text-slate-500">({filtered.length} of {hotspots.length})</span>
          </h2>
          <input
            type="text"
            aria-label="Search hotspots"
            placeholder="Search class, location, FRP, date..."
            className="text-xs border border-slate-300 dark:border-slate-600 rounded-lg px-2 py-1.5 bg-white dark:bg-slate-800 text-slate-700 dark:text-slate-200 w-56"
            value={query}
            onChange={(e) => {
              setQuery(e.target.value);
              setPage(0);
            }}
          />
          <select
            aria-label="Class filter"
            className="text-xs border border-slate-300 dark:border-slate-600 rounded-lg px-2 py-1.5 bg-white dark:bg-slate-800 text-slate-700 dark:text-slate-200"
            value={classFilter}
            onChange={(e) => {
              setClassFilter(e.target.value as ClassKey | 'all');
              setPage(0);
            }}
          >
            <option value="all">All classes</option>
            {ALL_CLASSES.map((c) => (
              <option key={c} value={c}>
                {CLASS_DISPLAY_NAMES[c]}
              </option>
            ))}
          </select>
          <select
            aria-label="Review filter"
            className="text-xs border border-slate-300 dark:border-slate-600 rounded-lg px-2 py-1.5 bg-white dark:bg-slate-800 text-slate-700 dark:text-slate-200"
            value={reviewFilter}
            onChange={(e) => {
              setReviewFilter(e.target.value as 'all' | 'auto' | 'flagged');
              setPage(0);
            }}
          >
            <option value="all">All</option>
            <option value="auto">Auto-classified</option>
            <option value="flagged">Flagged</option>
          </select>
          <button
            type="button"
            onClick={handleExport}
            className="text-xs font-semibold border border-slate-300 dark:border-slate-600 rounded-lg px-3 py-1.5 text-slate-700 dark:text-slate-200 hover:bg-slate-100 dark:hover:bg-slate-700"
          >
            Export CSV
          </button>
        </div>

        <div className="overflow-x-auto border border-slate-200 dark:border-slate-700 rounded-2xl bg-white dark:bg-slate-900 shadow">
          <table className="w-full text-xs">
            <thead>
              <tr className="bg-slate-50 dark:bg-slate-800">
                {COLUMNS.map((c) => (
                  <th key={c.key}>
                    <button
                      type="button"
                      onClick={() => clickSort(c.key)}
                      className="w-full text-left font-bold text-slate-600 dark:text-slate-300 px-3 py-2 hover:text-slate-900 dark:hover:text-white whitespace-nowrap"
                    >
                      {c.label} {sortKey === c.key ? (sortDir === 1 ? '▲' : '▼') : ''}
                    </button>
                  </th>
                ))}
                <th className="px-3 py-2 text-left font-bold text-slate-600 dark:text-slate-300">Map</th>
              </tr>
            </thead>
            <tbody>
              {pageRows.map((h) => {
                const idx = hotspots.indexOf(h);
                return (
                  <tr
                    key={h.id}
                    className="border-t border-slate-100 dark:border-slate-800 odd:bg-white even:bg-slate-50/60 dark:odd:bg-slate-900 dark:even:bg-slate-800/40 hover:bg-red-50 dark:hover:bg-slate-700/50 cursor-pointer"
                    onClick={() => onViewOnMap(idx)}
                    title="Open on map"
                  >
                    <td className="px-3 py-1.5 font-mono">{h.id}</td>
                    <td className="px-3 py-1.5">{h.lat.toFixed(3)}</td>
                    <td className="px-3 py-1.5">{h.lon.toFixed(3)}</td>
                    <td className="px-3 py-1.5 font-semibold">{h.frp}</td>
                    <td className="px-3 py-1.5">{h.bright_ti4}</td>
                    <td className="px-3 py-1.5 capitalize">{h.confidence}</td>
                    <td className="px-3 py-1.5">{h.daynight}</td>
                    <td className="px-3 py-1.5 whitespace-nowrap">{h.acq_date}</td>
                    <td className="px-3 py-1.5 whitespace-nowrap">{h.classification.display_name}</td>
                    <td className="px-3 py-1.5">{h.classification.decision_source}</td>
                    <td className="px-3 py-1.5">{(h.classification.confidence * 100).toFixed(1)}%</td>
                    <td className="px-3 py-1.5">{h.classification.requires_human_review ? '⚠ Yes' : 'No'}</td>
                    <td className="px-3 py-1.5">
                      <button
                        type="button"
                        className="font-semibold text-red-600 hover:text-red-700 whitespace-nowrap"
                        onClick={(e) => {
                          e.stopPropagation();
                          onViewOnMap(idx);
                        }}
                      >
                        View on Map
                      </button>
                    </td>
                  </tr>
                );
              })}
              {pageRows.length === 0 && (
                <tr>
                  <td colSpan={COLUMNS.length + 1} className="px-3 py-8 text-center text-slate-400">
                    No hotspots match the current search and filters.
                  </td>
                </tr>
              )}
            </tbody>
          </table>
        </div>

        <div className="flex items-center gap-3 mt-3 text-xs text-slate-600 dark:text-slate-300">
          <button
            type="button"
            disabled={safePage === 0}
            onClick={() => setPage((p) => Math.max(0, p - 1))}
            className="font-semibold border border-slate-300 dark:border-slate-600 rounded-lg px-3 py-1.5 disabled:opacity-40"
          >
            Prev
          </button>
          <span>
            Page {safePage + 1} of {pageCount}
          </span>
          <button
            type="button"
            disabled={safePage >= pageCount - 1}
            onClick={() => setPage((p) => Math.min(pageCount - 1, p + 1))}
            className="font-semibold border border-slate-300 dark:border-slate-600 rounded-lg px-3 py-1.5 disabled:opacity-40"
          >
            Next
          </button>
          <select
            aria-label="Rows per page"
            className="border border-slate-300 dark:border-slate-600 rounded-lg px-2 py-1.5 bg-white dark:bg-slate-800"
            value={perPage}
            onChange={(e) => {
              setPerPage(Number(e.target.value));
              setPage(0);
            }}
          >
            {[10, 25, 50].map((n) => (
              <option key={n} value={n}>
                {n} / page
              </option>
            ))}
          </select>
        </div>
      </div>
    </div>
  );
}
