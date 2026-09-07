import { useState } from 'react';
import { Hotspot } from '../types/hotspot';
import { useSearch } from '../hooks/useSearch';

interface Props {
  hotspots: Hotspot[];
  onFlyToCoords: (lat: number, lon: number) => void;
  onSelectNearest: (lat: number, lon: number) => void;
  showToast: (msg: string) => void;
}

export default function SearchBox({ hotspots, onFlyToCoords, onSelectNearest, showToast }: Props) {
  const [value, setValue] = useState('');
  const { status, run } = useSearch(hotspots, onFlyToCoords, onSelectNearest, showToast);

  return (
    <form
      className="flex items-center gap-1"
      onSubmit={(e) => {
        e.preventDefault();
        run(value);
      }}
    >
        <input
          type="text"
          aria-label="Search coordinates or place"
          placeholder="Search coordinates (e.g. 21.1051, 72.6438) or place name..."
          className="text-xs border border-slate-300 dark:border-slate-600 rounded-lg px-2 py-1.5 bg-white dark:bg-slate-800 text-slate-700 dark:text-slate-200 w-52 sm:w-64 transition-colors duration-150 ease-in-out focus:outline-none focus-visible:ring-2 focus-visible:ring-offset-2 focus-visible:ring-sky-400"
          value={value}
          onChange={(e) => setValue(e.target.value)}
        />
        <button
          type="submit"
          aria-label="Search"
          disabled={status === 'searching'}
          className="text-xs font-semibold border border-slate-300 dark:border-slate-600 rounded-lg px-2 py-1.5 text-slate-600 dark:text-slate-300 hover:bg-slate-100 dark:hover:bg-slate-700 disabled:opacity-50 transition-all duration-150 ease-in-out active:scale-[0.98] focus-visible:ring-2 focus-visible:ring-offset-2 focus-visible:ring-sky-400"
        >
          {status === 'searching' ? '…' : 'Go'}
        </button>
    </form>
  );
}
