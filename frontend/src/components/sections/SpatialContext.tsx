import { SpatialContext as SC } from '../../types/hotspot';

interface Props {
  spatial_context: SC;
}

const ROWS: { key: keyof SC; label: string }[] = [
  { key: 'industry', label: 'Industry' },
  { key: 'refinery', label: 'Refinery' },
  { key: 'oil_gas', label: 'Oil & Gas Site' },
  { key: 'mining', label: 'Mining Site' },
  { key: 'forest', label: 'Forest Preserve' },
  { key: 'agriculture', label: 'Agriculture Land' },
  { key: 'power_plant', label: 'Power Plant' },
];

export default function SpatialContext({ spatial_context }: Props) {
  return (
    <div>
      <h3 className="flex items-center gap-2 mb-3 text-xs font-semibold uppercase tracking-wider text-slate-500 dark:text-slate-400">
        <svg className="w-3 h-3 text-slate-400" fill="none" stroke="currentColor" viewBox="0 0 24 24" aria-hidden="true">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M17.657 16.657L13.414 20.9a1.998 1.998 0 01-2.827 0l-4.244-4.243a8 8 0 1111.314 0zM15 11a3 3 0 11-6 0 3 3 0 016 0z" />
        </svg>
        Nearest Infrastructure (within 15 km)
      </h3>
      <div>
        {ROWS.map((r) => {
          const val = spatial_context[r.key];
          return (
            <div key={r.key} className="flex justify-between items-center gap-2 py-2 border-b border-slate-100 dark:border-slate-700 last:border-0 text-xs">
              <span className="text-slate-500 dark:text-slate-400 font-medium">{r.label}</span>
              {val === null ? (
                <span className="text-slate-400">none within 15 km</span>
              ) : (
                <span className="font-mono text-sm font-medium text-slate-800 dark:text-slate-200">{val.toLocaleString()} m</span>
              )}
            </div>
          );
        })}
      </div>
    </div>
  );
}
