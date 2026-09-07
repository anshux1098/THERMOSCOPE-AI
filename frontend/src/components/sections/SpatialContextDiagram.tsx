import { SpatialContext as SC } from '../../types/hotspot';

interface Props {
  spatial_context: SC;
}

const ROWS: { key: keyof SC; label: string; color: string }[] = [
  { key: 'industry', label: 'Industry', color: '#e53935' },
  { key: 'refinery', label: 'Refinery', color: '#fb8c00' },
  { key: 'oil_gas', label: 'Oil & Gas', color: '#fdd835' },
  { key: 'mining', label: 'Mining', color: '#8d6e63' },
  { key: 'forest', label: 'Forest', color: '#2e7d32' },
  { key: 'agriculture', label: 'Agriculture', color: '#689f38' },
  { key: 'power_plant', label: 'Power Plant', color: '#8e24aa' },
];

const MAX_M = 15000;

export default function SpatialContextDiagram({ spatial_context }: Props) {
  const present = ROWS.map((r) => ({ ...r, dist: spatial_context[r.key] })).filter(
    (r) => r.dist !== null,
  ) as { key: keyof SC; label: string; color: string; dist: number }[];
  const sorted = [...present].sort((a, b) => a.dist - b.dist);

  return (
    <div>
      <h3 className="flex items-center gap-2 mb-3 text-xs font-semibold uppercase tracking-wider text-slate-500 dark:text-slate-400">
        <svg className="w-3 h-3 text-slate-400" fill="none" stroke="currentColor" viewBox="0 0 24 24" aria-hidden="true">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M17.657 16.657L13.414 20.9a1.998 1.998 0 01-2.827 0l-4.244-4.243a8 8 0 1111.314 0zM15 11a3 3 0 11-6 0 3 3 0 016 0z" />
        </svg>
        Spatial Evidence at a Glance
      </h3>
      <div>
        {sorted.length === 0 ? (
          <p className="text-xs text-slate-400">No OSM infrastructure within 15 km of this hotspot.</p>
        ) : (
          <div className="space-y-2">
            {sorted.map((r) => (
              <div key={r.key}>
                <div className="flex justify-between text-xs">
                  <span className="text-slate-500 dark:text-slate-400 font-medium">{r.label}</span>
                  <span className="font-mono text-sm font-medium text-slate-800 dark:text-slate-200">{r.dist.toLocaleString()} m</span>
                </div>
                <div className="w-full bg-slate-100 dark:bg-slate-700 h-2 rounded-full overflow-hidden mt-0.5">
                  <div
                    className="h-full rounded-full"
                    style={{
                      width: `${Math.min(100, (r.dist / MAX_M) * 100)}%`,
                      backgroundColor: r.color,
                    }}
                  />
                </div>
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}
