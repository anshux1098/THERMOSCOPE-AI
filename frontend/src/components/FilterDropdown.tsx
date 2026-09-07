import { useEffect, useRef, useState } from 'react';
import { ClassKey, CLASS_COLORS, CLASS_DISPLAY_NAMES } from '../types/hotspot';
import { ALL_CLASSES } from '../hooks/useAppState';

interface Props {
  activeClasses: ClassKey[];
  toggleClass: (c: ClassKey) => void;
  selectAllClasses: (on: boolean) => void;
  /** Panel anchoring inside narrow containers (default 'left'). */
  panelAlign?: 'left' | 'right';
}

const ORDER: ClassKey[] = [
  'industrial_fire',
  'gas_flare',
  'mining_activity',
  'agricultural_burn',
  'forest_natural_fire',
  'industrial_process_heat',
  'unclassified',
];

function GroupLabel({ children }: { children: React.ReactNode }) {
  return (
    <div className="text-xs font-medium uppercase tracking-wider text-slate-400 dark:text-slate-500 mt-3 mb-1.5 first:mt-0">
      {children}
    </div>
  );
}

export default function FilterDropdown(props: Props) {
  const {
    activeClasses,
    toggleClass,
    selectAllClasses,
    panelAlign = 'left',
  } = props;
  const [open, setOpen] = useState(false);
  const boxRef = useRef<HTMLDivElement>(null);

  const allOn = activeClasses.length === ALL_CLASSES.length;
  const label = allOn ? 'All Classes' : `${activeClasses.length} Classes Selected`;

  useEffect(() => {
    if (!open) return;
    const onDown = (e: MouseEvent) => {
      if (boxRef.current && !boxRef.current.contains(e.target as Node)) setOpen(false);
    };
    const onKey = (e: KeyboardEvent) => {
      if (e.key === 'Escape') setOpen(false);
    };
    document.addEventListener('mousedown', onDown);
    document.addEventListener('keydown', onKey);
    return () => {
      document.removeEventListener('mousedown', onDown);
      document.removeEventListener('keydown', onKey);
    };
  }, [open ]);

  return (
    <div ref={boxRef} className="relative">
      <button
        type="button"
        aria-expanded={open}
        aria-haspopup="true"
        onClick={() => setOpen((v) => !v)}
        className={`inline-flex items-center gap-2 px-4 py-2 text-sm font-medium rounded-full border cursor-pointer transition-all duration-150 ease-in-out focus:ring-2 focus:ring-amber-400 focus:border-transparent focus:outline-none ${
          open
            ? 'bg-slate-100 dark:bg-slate-700 border-slate-300 dark:border-slate-500 text-slate-900 dark:text-slate-50'
            : 'bg-white dark:bg-slate-800 border-slate-200 dark:border-slate-700 text-slate-700 dark:text-slate-50 hover:bg-slate-100 dark:hover:bg-slate-700'
        }`}
      >
        <svg className="w-4 h-4 text-slate-400" fill="none" stroke="currentColor" viewBox="0 0 24 24" aria-hidden="true">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M3 4a1 1 0 011-1h16a1 1 0 011 1v2.586a1 1 0 01-.293.707l-6.414 6.414a1 1 0 00-.293.707V17l-4 4v-6.586a1 1 0 00-.293-.707L3.293 7.293A1 1 0 013 6.586V4z" />
        </svg>
        {label}
        <svg className={`w-3.5 h-3.5 text-slate-400 transition-transform duration-200 ${open ? 'rotate-180' : ''}`} fill="none" stroke="currentColor" viewBox="0 0 24 24" aria-hidden="true">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
        </svg>
      </button>

      {open && (
        <div className={`absolute top-full mt-1 left-0 right-0 max-h-80 overflow-y-auto bg-white dark:bg-slate-800 border border-slate-200 dark:border-slate-700 border-t-0 rounded-b-xl shadow-lg z-50 p-4 ${panelAlign === 'right' ? 'right-0 left-auto min-w-[220px]' : ''}`}>
          <GroupLabel>Filter by Class</GroupLabel>
          <label className="flex items-center gap-2.5 py-1.5 cursor-pointer rounded-lg px-1 hover:bg-slate-50 dark:hover:bg-slate-700 min-h-[44px]">
            <input
              type="checkbox"
              checked={allOn}
              onChange={(e) => selectAllClasses(e.target.checked)}
              className="w-3.5 h-3.5 rounded accent-amber-500 focus-visible:ring-2 focus-visible:ring-amber-400"
            />
            <span className="text-sm font-semibold text-slate-800 dark:text-slate-100">All Classes</span>
          </label>
          <div className="mt-1 space-y-0.5">
            {ORDER.map((c) => {
              const on = activeClasses.includes(c);
              return (
                <label
                  key={c}
                  className="flex items-center gap-2.5 py-1.5 cursor-pointer rounded-lg px-1 hover:bg-slate-50 dark:hover:bg-slate-700 min-h-[44px]"
                >
                  <input
                    type="checkbox"
                    checked={on}
                    onChange={() => toggleClass(c)}
                    className="w-3.5 h-3.5 rounded accent-amber-500 focus-visible:ring-2 focus-visible:ring-amber-400"
                  />
                  <span className="w-3 h-3 rounded-full shrink-0" style={{ backgroundColor: CLASS_COLORS[c] }} />
                  <span className="text-sm text-slate-700 dark:text-slate-50">{CLASS_DISPLAY_NAMES[c]}</span>
                </label>
              );
            })}
          </div>
        </div>
      )}
    </div>
  );
}
