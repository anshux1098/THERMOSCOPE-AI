export default function Header() {

  return (
    <div className="flex items-center justify-between px-4 py-2 bg-white dark:bg-slate-900 border-b border-slate-200 dark:border-slate-700 z-40">
      <div className="flex items-center gap-2 min-w-0">
        <svg viewBox="0 0 24 24" className="w-6 h-6 text-amber-500 shrink-0" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
          <path d="M12 3c1.5 3-3 4.5-3 8a3 3 0 006 0c0-1.2-.4-1.9-.8-2.7-.6.6-.8 1.2-.8 2 0-2.2-1-4-1.4-5.3z" />
          <path d="M12 22a5 5 0 01-5-5c0-.4.1-.8.2-1.2C8 17 9 18 12 18s4-1 4.8-2.2c.1.4.2.8.2 1.2a5 5 0 01-5 5z" />
        </svg>
        <div className="leading-tight">
          <h1 className="flex items-baseline gap-1 font-bold text-lg tracking-tight">
            <span className="text-slate-900 dark:text-slate-50">THERMOSCOPE</span>
            <span className="text-amber-500 italic">AI</span>
          </h1>
          <p className="text-[11px] text-slate-500 dark:text-slate-400">Industrial Fire Intelligence Platform</p>
        </div>
      </div>
      <div className="text-[11px] text-slate-400 dark:text-slate-500 hidden md:block">SIH 2026 · NTRO</div>
    </div>
  );
}
