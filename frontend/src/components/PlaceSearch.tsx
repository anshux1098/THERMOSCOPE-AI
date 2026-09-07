import { useEffect, useRef, useState } from 'react';
import { Hotspot } from '../types/hotspot';
import { calculateRateLimitDelay, NominatimResult, searchPlace } from '../services/nominatim';

interface Props {
  hotspots: Hotspot[];
  onFlyToCoords: (lat: number, lon: number) => void;
  onSelectNearest: (lat: number, lon: number) => void;
  onPlacePin: (lat: number, lon: number, label: string) => void;
  onClearPins: () => void;
  showToast: (msg: string, type?: 'info' | 'success' | 'warning' | 'error') => void;
}

export default function PlaceSearch({ onFlyToCoords, onSelectNearest, onPlacePin, onClearPins, showToast }: Props) {
  const [value, setValue] = useState('');
  const [results, setResults] = useState<NominatimResult[]>([]);
  const [open, setOpen] = useState(false);
  const [searching, setSearching] = useState(false);
  const debounce = useRef<number | null>(null);
  const abortRef = useRef<AbortController | null>(null);
  const lastCall = useRef<number | null>(null);
  const boxRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const onDocClick = (e: MouseEvent) => {
      if (boxRef.current && !boxRef.current.contains(e.target as Node)) setOpen(false);
    };
    document.addEventListener('mousedown', onDocClick);
    return () => document.removeEventListener('mousedown', onDocClick);
  }, []);

  useEffect(
    () => () => {
      if (debounce.current !== null) window.clearTimeout(debounce.current);
      abortRef.current?.abort();
    },
    [],
  );

  const runQuery = (q: string) => {
    if (debounce.current !== null) window.clearTimeout(debounce.current);
    debounce.current = window.setTimeout(() => void doSearch(q.trim()), 300);
  };

  const doSearch = async (q: string) => {
    if (!q) {
      setResults([]);
      setOpen(false);
      return;
    }
    const coord = q.match(/^(-?\d+(?:\.\d+)?)[,\s]+(-?\d+(?:\.\d+)?)$/);
    if (coord) {
      const lat = parseFloat(coord[1]);
      const lon = parseFloat(coord[2]);
      if (!Number.isFinite(lat) || !Number.isFinite(lon) || Math.abs(lat) > 90 || Math.abs(lon) > 180) {
        showToast('Invalid coordinates. Use format: lat, lon', 'warning');
        return;
      }
      setResults([]);
      setOpen(false);
      onFlyToCoords(lat, lon);
      onSelectNearest(lat, lon);
      return;
    }
    const delay = calculateRateLimitDelay(lastCall.current);
    if (delay > 0) {
      showToast('Please wait before searching again', 'warning');
      return;
    }
    abortRef.current?.abort();
    abortRef.current = new AbortController();
    setSearching(true);
    try {
      lastCall.current = Date.now();
      const data = await searchPlace(q, abortRef.current.signal);
      if (data.length === 0) {
        showToast(`No places found for '${q}'`, 'warning');
        setResults([]);
        setOpen(false);
        return;
      }
      setResults(data);
      setOpen(true);
    } catch (err) {
      if ((err as Error)?.name !== 'AbortError') showToast('Geocoding service unavailable', 'error');
    } finally {
      setSearching(false);
    }
  };

  const pick = (r: NominatimResult) => {
    const lat = parseFloat(r.lat);
    const lon = parseFloat(r.lon);
    setOpen(false);
    onFlyToCoords(lat, lon);
    onPlacePin(lat, lon, r.display_name);
    showToast(`Found: ${r.display_name.split(',').slice(0, 2).join(',')}`, 'success');
  };

  const clear = () => {
    setValue('');
    setResults([]);
    setOpen(false);
    onClearPins();
  };

  return (
    <div ref={boxRef} className="relative">
      <form
        className="flex items-center gap-1"
        onSubmit={(e) => {
          e.preventDefault();
          runQuery(value);
        }}
      >
        <input
          type="text"
          aria-label="Search coordinates or place"
          placeholder="Search coordinates (e.g. 21.1051, 72.6438) or place name..."
          className="text-xs border border-slate-300 dark:border-slate-600 rounded-lg px-2 py-1.5 bg-white dark:bg-slate-800 text-slate-700 dark:text-slate-200 w-52 sm:w-64 transition-colors duration-150 ease-in-out focus:outline-none focus-visible:ring-2 focus-visible:ring-offset-2 focus-visible:ring-sky-400"
          value={value}
          onChange={(e) => {
            setValue(e.target.value);
            if (e.target.value.trim().length >= 3) runQuery(e.target.value);
            else {
              setResults([]);
              setOpen(false);
            }
          }}
        />
        <button
          type="submit"
          aria-label="Search"
          disabled={searching}
          className="text-xs font-semibold border border-slate-300 dark:border-slate-600 rounded-lg px-2 py-1.5 text-slate-600 dark:text-slate-300 hover:bg-slate-100 dark:hover:bg-slate-700 disabled:opacity-50 transition-all duration-150 ease-in-out active:scale-[0.98] focus-visible:ring-2 focus-visible:ring-offset-2 focus-visible:ring-sky-400"
        >
          {searching ? '…' : 'Go'}
        </button>
        {(value || results.length > 0) && (
          <button
            type="button"
            aria-label="Clear search"
            onClick={clear}
            className="text-xs font-semibold border border-slate-300 dark:border-slate-600 rounded-lg px-2 py-1.5 text-slate-600 dark:text-slate-300 hover:bg-slate-100 dark:hover:bg-slate-700 transition-all duration-150 ease-in-out active:scale-[0.98] focus-visible:ring-2 focus-visible:ring-offset-2 focus-visible:ring-sky-400"
          >
            Clear
          </button>
        )}
      </form>

      {open && results.length > 0 && (
        <ul className="absolute top-full mt-1 left-0 right-0 bg-white dark:bg-slate-800 border border-slate-300 dark:border-slate-600 rounded-xl shadow-xl overflow-hidden z-30">
          {results.map((r) => (
            <li key={r.place_id}>
              <button
                type="button"
                className="w-full text-left px-3 py-2 hover:bg-slate-100 dark:hover:bg-slate-700 flex items-center gap-2"
                onClick={() => pick(r)}
              >
                <span aria-hidden="true">📍</span>
                <span className="min-w-0">
                  <span className="block text-xs text-slate-800 dark:text-slate-200 truncate">
                    {r.display_name.length > 60 ? `${r.display_name.slice(0, 60)}…` : r.display_name}
                  </span>
                  <span className="block text-[10px] text-slate-400">
                    {r.type} · {r.class}
                  </span>
                </span>
              </button>
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}
