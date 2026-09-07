import { useEffect, useRef, useState } from 'react';
import { ConfFilter, ReviewFilter } from '../hooks/useAppState';

interface Props {
  confFilter: ConfFilter;
  reviewFilter: ReviewFilter;
  onlyClassified: boolean;
  setConfFilter: (c: ConfFilter) => void;
  setReviewFilter: (r: ReviewFilter) => void;
  toggleOnlyClassified: () => void;
  panelAlign?: 'left' | 'right';
}

const CONF_OPTS: ConfFilter[] = ['All', 'High', 'Medium', 'Low'];
const REVIEW_OPTS: { v: ReviewFilter; l: string }[] = [
  { v: 'all', l: 'All' },
  { v: 'auto', l: 'Auto-Classified' },
  { v: 'flagged', l: 'Flagged for Review' },
];

function GroupLabel({ children }: { children: React.ReactNode }) {
  return (
    <div className="text-xs font-medium uppercase tracking-wider text-slate-400 dark:text-slate-500 mt-3 mb-1.5 first:mt-0">
      {children}
    </div>
  );
}

const CONF_SUMMARY: Record<ConfFilter, string> = {
  All: 'Any confidence',
  High: 'High confidence',
  Medium: 'Medium confidence',
  Low: 'Low confidence',
};

const REVIEW_SUMMARY: Record<ReviewFilter, string> = {
  all: 'All + flagged',
  auto: 'Auto-classified',
  flagged: 'Flagged only',
};

export default function ViewOptionsDropdown(props: Props) {
  const {
    confFilter,
    reviewFilter,
    onlyClassified,
    setConfFilter,
    setReviewFilter,
    toggleOnlyClassified,
    panelAlign = 'left',
  } = props;
  const [open, setOpen] = useState(false);
  const boxRef = useRef<HTMLDivElement>(null);

  const isDefault = confFilter === 'All' && reviewFilter === 'all' && !onlyClassified;
  const label = isDefault ? 'View' : `${CONF_SUMMARY[confFilter]} · ${REVIEW_SUMMARY[reviewFilter]}${onlyClassified ? ' · Classified' : ''}`;

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
  }, [open]);

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
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 12a3 3 0 11-6 0 3 3 0 016 0z" />
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M2.458 12C3.732 7.943 7.523 5 12 5c4.478 0 8.268 2.943 9.542 7-1.274 4.057-5.064 7-9.542 7-4.477 0-8.268-2.943-9.542-7z" />
        </svg>
        <span className="truncate max-w-[180px]">{label}</span>
        {!isDefault && <span className="w-2 h-2 rounded-full bg-amber-500 shrink-0" aria-label="view options active" />}
        <svg className={`w-3.5 h-3.5 text-slate-400 transition-transform duration-200 ${open ? 'rotate-180' : ''}`} fill="none" stroke="currentColor" viewBox="0 0 24 24" aria-hidden="true">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
        </svg>
      </button>

      {open && (
        <div className={`absolute top-full mt-1 w-64 max-h-80 overflow-y-auto bg-white dark:bg-slate-800 border border-slate-200 dark:border-slate-700 border-t-0 rounded-b-xl shadow-lg z-50 p-4 ${panelAlign === 'right' ? 'right-0' : 'left-0'}`}>
          <GroupLabel>Confidence</GroupLabel>
          <div className="space-y-0.5">
            {CONF_OPTS.map((c) => (
              <label key={c} className="flex items-center gap-2.5 py-1.5 cursor-pointer rounded-lg px-1 hover:bg-slate-50 dark:hover:bg-slate-700 min-h-[44px]">
                <input
                  type="radio"
                  name="view-conf"
                  checked={confFilter === c}
                  onChange={() => setConfFilter(c)}
                  className="w-3.5 h-3.5 accent-amber-500 focus-visible:ring-2 focus-visible:ring-amber-400"
                />
                <span className="text-sm text-slate-700 dark:text-slate-50">{c === 'All' ? 'All levels' : c}</span>
              </label>
            ))}
          </div>

          <GroupLabel>Review Status</GroupLabel>
          <div className="space-y-0.5">
            {REVIEW_OPTS.map((o) => (
              <label key={o.v} className="flex items-center gap-2.5 py-1.5 cursor-pointer rounded-lg px-1 hover:bg-slate-50 dark:hover:bg-slate-700 min-h-[44px]">
                <input
                  type="radio"
                  name="view-review"
                  checked={reviewFilter === o.v}
                  onChange={() => setReviewFilter(o.v)}
                  className="w-3.5 h-3.5 accent-amber-500 focus-visible:ring-2 focus-visible:ring-amber-400"
                />
                <span className="text-sm text-slate-700 dark:text-slate-50">{o.l}</span>
              </label>
            ))}
          </div>

          <GroupLabel>Options</GroupLabel>
          <label className="flex items-center gap-2.5 py-1.5 cursor-pointer rounded-lg px-1 hover:bg-slate-50 dark:hover:bg-slate-700 min-h-[44px]">
            <span
              role="checkbox"
              aria-checked={onlyClassified}
              tabIndex={0}
              onClick={toggleOnlyClassified}
              onKeyDown={(e) => {
                if (e.key === ' ' || e.key === 'Enter') {
                  e.preventDefault();
                  toggleOnlyClassified();
                }
              }}
              className={`relative inline-flex w-9 h-5 rounded-full transition-colors duration-200 focus-visible:ring-2 focus-visible:ring-amber-400 focus:outline-none ${onlyClassified ? 'bg-amber-500' : 'bg-slate-300 dark:bg-slate-600'}`}
            >
              <span
                className={`inline-block w-3.5 h-3.5 rounded-full bg-white shadow-sm transform transition-transform duration-200 mt-0.5 ${onlyClassified ? 'translate-x-4 ml-0.5' : 'translate-x-0 ml-0.5'}`}
              />
            </span>
            <input type="checkbox" className="sr-only" checked={onlyClassified} onChange={toggleOnlyClassified} />
            <span className="text-sm text-slate-700 dark:text-slate-50">Classified Only</span>
          </label>
        </div>
      )}
    </div>
  );
}
