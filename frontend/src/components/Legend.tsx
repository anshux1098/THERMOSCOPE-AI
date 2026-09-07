import { ClassKey, CLASS_COLORS, CLASS_DISPLAY_NAMES, Hotspot } from '../types/hotspot';

interface LegendProps {
  filterClass: ClassKey | 'all';
  setFilterClass: (c: ClassKey | 'all') => void;
  hotspots: Hotspot[];
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

export default function Legend({ filterClass, setFilterClass, hotspots }: LegendProps) {
  const counts = (() => {
    const m = new Map<ClassKey, number>();
    for (const h of hotspots) {
      const c = h.classification.class as ClassKey;
      m.set(c, (m.get(c) ?? 0) + 1);
    }
    return m;
  })();

  return (
    <div className="bg-white/95 dark:bg-slate-800/95 backdrop-blur border border-slate-200 dark:border-slate-700 rounded-full shadow-[0_2px_8px_-2px_rgba(0,0,0,0.06)] dark:shadow-[0_2px_8px_-2px_rgba(0,0,0,0.3)] px-3 py-1">
      <div className="flex items-center gap-x-1.5 gap-y-0.5 overflow-x-auto whitespace-nowrap max-w-[70vw]">
        <span className="text-[10px] font-bold text-slate-400 dark:text-slate-500 uppercase tracking-wider shrink-0" title="Click a class to filter">
          Legend
        </span>
        {ORDER.map((cls) => {
          const active = filterClass === cls;
          const muted = cls === 'unclassified' && active;
          const n = counts.get(cls) ?? 0;
          const dimmed = filterClass !== 'all' && !active;
          return (
            <button
              key={cls}
              type="button"
              aria-label={`Filter ${CLASS_DISPLAY_NAMES[cls]} (${n})`}
              aria-pressed={active}
              title="Click to filter"
              className={`flex items-center gap-1 text-[11px] font-medium px-1.5 py-0.5 rounded-md transition-all duration-150 ease-in-out hover:scale-[1.03] active:scale-[0.98] hover:bg-slate-100 dark:hover:bg-slate-700 focus-visible:ring-2 focus-visible:ring-offset-2 focus-visible:ring-sky-400 shrink-0 ${
                active ? 'ring-2 ring-offset-2 ring-slate-400 bg-slate-100 dark:bg-slate-700' : 'text-slate-600 dark:text-slate-300'
              } ${muted ? 'opacity-80' : ''} ${dimmed ? 'opacity-50' : ''}`}
              onClick={() => setFilterClass(active ? 'all' : cls)}
            >
              <span
                className="rounded-full inline-block"
                style={{ width: 10, height: 10, backgroundColor: CLASS_COLORS[cls] }}
              />
              {CLASS_DISPLAY_NAMES[cls]}
              <span className="text-[10px] font-semibold text-slate-400 dark:text-slate-500">{n}</span>
            </button>
          );
        })}
      </div>
    </div>
  );
}
