import { useState } from 'react';
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

export default function MlProbabilities({ ml_engine }: Props) {
  const [open, setOpen] = useState(false);

  return (
    <div>
      <button
        type="button"
        aria-expanded={open}
        onClick={() => setOpen((v) => !v)}
        className="w-full p-3 bg-slate-50 hover:bg-slate-100 flex items-center justify-between text-left font-bold text-slate-800 text-sm rounded-xl border border-slate-200"
      >
        <span>XGBoost ML Probabilities (7 Classes)</span>
        <svg
          viewBox="0 0 24 24"
          className={`w-4 h-4 text-slate-500 transition-transform ${open ? 'rotate-180' : ''}`}
          fill="none"
          stroke="currentColor"
          strokeWidth={2}
          aria-hidden="true"
        >
          <path strokeLinecap="round" strokeLinejoin="round" d="M19 9l-7 7-7-7" />
        </svg>
      </button>
      <div
        className={`overflow-hidden transition-all ease-out ${
          open ? 'max-h-96 opacity-100' : 'max-h-0 opacity-0'
        }`}
        style={{ transitionDuration: '0.25s' }}
      >
        <div className="space-y-2.5 mt-2 px-1 pb-1">
          {ORDER.map((cls) => {
            const prob = ml_engine.probabilities[cls] ?? 0;
            const pct = (prob * 100).toFixed(1);
            const isPred = cls === ml_engine.predicted_class;
            return (
              <div key={cls}>
                <div
                  className={`flex justify-between text-[11px] ${
                    isPred ? 'font-bold text-slate-900' : 'text-slate-600'
                  }`}
                >
                  <span>
                    {cls.replace(/_/g, ' ')}
                    {isPred && ' (Predicted)'}
                  </span>
                  <span>{pct}%</span>
                </div>
                <div className="w-full bg-slate-100 h-2 rounded-full overflow-hidden mt-0.5">
                  <div
                    className="h-full rounded-full"
                    style={{ width: `${pct}%`, backgroundColor: CLASS_COLORS[cls] }}
                  />
                </div>
              </div>
            );
          })}
        </div>
      </div>
    </div>
  );
}
