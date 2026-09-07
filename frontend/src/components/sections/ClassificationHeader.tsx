import { ClassKey, CLASS_COLORS, Classification } from '../../types/hotspot';

interface Props {
  classification: Classification;
}

const LEVEL_STYLE: Record<string, { badge: string; bar: string }> = {
  High: { badge: 'bg-emerald-100 text-emerald-800', bar: 'bg-gradient-to-r from-emerald-600 to-emerald-400' },
  Medium: { badge: 'bg-amber-100 text-amber-800', bar: 'bg-gradient-to-r from-amber-600 to-amber-400' },
  Low: { badge: 'bg-slate-200 text-slate-600', bar: 'bg-gradient-to-r from-slate-500 to-slate-400' },
};

export default function ClassificationHeader({ classification }: Props) {
  const color = CLASS_COLORS[classification.class as ClassKey] ?? '#9e9e9e';
  const level = LEVEL_STYLE[classification.confidence_level] ?? LEVEL_STYLE.Low;
  const pct = (classification.confidence * 100).toFixed(1);

  return (
    <div>
      <div className="flex items-center gap-2">
        <span className="w-4 h-4 rounded-full inline-block" style={{ backgroundColor: color }} />
        <span className="font-bold text-slate-900 dark:text-slate-100 text-base">{classification.display_name}</span>
      </div>

      <div className="flex flex-wrap items-center gap-1.5 mt-2">
        <span className="text-[11px] font-semibold px-2 py-0.5 rounded-full bg-slate-100 text-slate-600">
          {classification.decision_source}
        </span>
        <span className="text-[11px] font-semibold px-2 py-0.5 rounded-full bg-red-50 text-red-700">
          {classification.risk_level} risk
        </span>
      </div>

      <div className="flex items-center gap-2 mt-2">
        <div className="w-16 h-1.5 rounded-full bg-slate-100 dark:bg-slate-700 overflow-hidden">
          <div className={`h-full rounded-full ${level.bar}`} style={{ width: `${pct}%` }} />
        </div>
        <span className="font-mono text-sm font-medium text-slate-800 dark:text-slate-200">{pct}%</span>
      </div>

      {classification.requires_human_review && (
        <div className="mt-2.5 bg-amber-50 border border-amber-200 p-2.5 rounded-xl text-amber-800 flex items-start space-x-2 text-xs">
          <span aria-hidden="true">⚠</span>
          <span>
            <strong>Requires human review:</strong> {classification.review_reason ?? 'Low confidence case.'}
          </span>
        </div>
      )}
    </div>
  );
}
