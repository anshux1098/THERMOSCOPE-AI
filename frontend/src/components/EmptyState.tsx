interface Props {
  visible: boolean;
  onClear: () => void;
}

export default function EmptyState({ visible, onClear }: Props) {
  if (!visible) return null;
  return (
    <div className="absolute inset-0 z-[13] flex items-center justify-center pointer-events-none p-4">
      <div className="bg-white dark:bg-slate-800 rounded-2xl shadow-xl border border-slate-200 dark:border-slate-700 px-6 py-5 text-center max-w-sm pointer-events-auto">
        <svg viewBox="0 0 24 24" className="w-12 h-12 mx-auto mb-3 text-slate-300" fill="none" stroke="currentColor" strokeWidth={1.5} aria-hidden="true">
          <circle cx="12" cy="12" r="9" />
          <path strokeLinecap="round" d="M15.5 8.5l-2 5-5 2 2-5z" />
        </svg>
        <p className="text-sm text-slate-500 dark:text-slate-400">
          No hotspots match the current filters. Try adjusting filters.
        </p>
        <button
          type="button"
          className="mt-3 text-xs font-semibold text-red-600 hover:text-red-700 border border-slate-300 dark:border-slate-600 rounded-lg px-3 py-1.5 transition-all duration-150 ease-in-out hover:scale-[1.02] active:scale-[0.98] focus-visible:ring-2 focus-visible:ring-offset-2 focus-visible:ring-sky-400"
          onClick={onClear}
        >
          Clear all filters
        </button>
      </div>
    </div>
  );
}
