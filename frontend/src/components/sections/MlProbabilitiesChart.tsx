import { ClassKey, CLASS_COLORS, MlEngine } from '../../types/hotspot';

interface Props {
  ml_engine: MlEngine;
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

export default function MlProbabilitiesChart({ ml_engine }: Props) {
  const sorted = [...ORDER].sort(
    (a, b) => (ml_engine.probabilities[b] ?? 0) - (ml_engine.probabilities[a] ?? 0),
  );

  return (
    <div>
      <h3 className="flex items-center gap-2 mb-3 text-xs font-semibold uppercase tracking-wider text-slate-500 dark:text-slate-400">
        <svg className="w-3 h-3 text-slate-400" fill="none" stroke="currentColor" viewBox="0 0 24 24" aria-hidden="true">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 20V10m0 0l-4-4m4 4l4-4m-4 4v10" />
        </svg>
        XGBoost ML Probabilities (7 Classes)
      </h3>
      <div className="space-y-2.5">
        {sorted.map((cls) => {
          const prob = ml_engine.probabilities[cls] ?? 0;
          const pct = (prob * 100).toFixed(1);
          const isPred = cls === ml_engine.predicted_class;
          return (
            <div key={cls} className="relative">
              <div
                className={`flex justify-between text-[11px] ${
                  isPred ? 'font-bold text-slate-900 dark:text-slate-100' : 'text-slate-600 dark:text-slate-300'
                }`}
              >
                <span>
                  {cls.replace(/_/g, ' ')}
                  {isPred && ' (Predicted)'}
                </span>
                <span className="font-mono">{pct}%</span>
              </div>
              <div className="w-full bg-slate-100 dark:bg-slate-700 h-2 rounded-full overflow-hidden mt-0.5">
                <div
                  className={`h-full rounded-full ${isPred ? 'ring-1 ring-offset-1 ring-slate-400' : ''}`}
                  style={{
                    width: `${pct}%`,
                    backgroundColor: CLASS_COLORS[cls],
                    boxShadow: isPred ? '0 0 6px rgba(0,0,0,0.35)' : undefined,
                  }}
                />
              </div>
              {isPred && (
                <div
                  className="absolute top-0 bottom-0 w-px bg-slate-900 dark:bg-white"
                  style={{ left: `${pct}%` }}
                  aria-hidden="true"
                />
              )}
            </div>
          );
        })}
      </div>
    </div>
  );
}
