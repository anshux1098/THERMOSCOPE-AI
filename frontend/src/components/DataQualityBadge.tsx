import { Hotspot } from '../types/hotspot';

interface Props {
  hotspot: Hotspot;
}

export default function DataQualityBadge({ hotspot }: Props) {
  const { confidence, daynight, satellite, acq_date } = hotspot;
  const quality = confidence === 'high' && daynight === 'D' ? 'Good' : confidence === 'low' ? 'Poor' : 'Fair';
  const style =
    quality === 'Good'
      ? 'bg-emerald-100 text-emerald-800 border-emerald-200'
      : quality === 'Fair'
        ? 'bg-amber-100 text-amber-800 border-amber-200'
        : 'bg-red-100 text-red-800 border-red-200';

  return (
    <div>
      <h3 className="flex items-center gap-2 mb-3 text-xs font-semibold uppercase tracking-wider text-slate-500 dark:text-slate-400">
        <svg className="w-3 h-3 text-slate-400" fill="none" stroke="currentColor" viewBox="0 0 24 24" aria-hidden="true">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z" />
        </svg>
        Data Quality
      </h3>
      <div className={`p-3 rounded-xl border text-xs ${style}`}>
        <div className="font-bold mb-1">Data Quality: {quality}</div>
        <div className="space-y-0.5 font-medium">
          <div>FIRMS confidence: {confidence}</div>
          <div>Day/Night: {daynight === 'N' ? 'Night (thermal-only)' : 'Day (visible spectrum)'}</div>
          <div>Satellite: {satellite}</div>
          <div>Acquisition: {acq_date}</div>
        </div>
      </div>
    </div>
  );
}
