import { PageKey } from '../hooks/useAppState';
import DarkModeToggle from './DarkModeToggle';

interface Props {
  activePage: PageKey;
  setActivePage: (p: PageKey) => void;
  alertCount: number;
}

const LINKS: { key: PageKey; label: string }[] = [
  { key: 'dashboard', label: 'Dashboard' },
  { key: 'analytics', label: 'Analytics' },
  { key: 'data-table', label: 'Data Table' },
  { key: 'methodology', label: 'Methodology' },
  { key: 'alerts', label: 'Alerts' },
];

export default function NavBar({ activePage, setActivePage, alertCount }: Props) {
  return (
    <nav
      aria-label="Primary"
      className="bg-white dark:bg-slate-900 border-b border-slate-200 dark:border-slate-700 z-40 relative"
    >
      <div className="flex items-center gap-1 px-3 py-1.5 overflow-x-auto">
        <span className="flex items-center gap-1.5 mr-2 shrink-0">
          <svg viewBox="0 0 24 24" className="w-5 h-5 text-red-600" fill="currentColor" aria-hidden="true">
            <path d="M12 2c1 4-4 5.5-4 10a4.5 4.5 0 009 0c0-1.5-.5-2.5-1-3.5-.8.8-1 1.5-1 2.5C14.5 7 13 4 12 2zm0 20a6.5 6.5 0 01-6.5-6.5c0-.5.1-1 .2-1.5C7 15.5 8.5 17 12 17s5-1.5 6.3-3c.1.5.2 1 .2 1.5A6.5 6.5 0 0112 22z" />
          </svg>
          <span className="font-bold text-slate-900 dark:text-slate-100 text-sm tracking-tight hidden sm:inline">
            THERMOSCOPE-AI
          </span>
        </span>
        {LINKS.map((l) => {
          const active = activePage === l.key;
          return (
            <button
              key={l.key}
              type="button"
              aria-current={active ? 'page' : undefined}
              onClick={() => setActivePage(l.key)}
              className={`inline-flex items-center text-xs font-semibold px-3 py-1.5 rounded-full whitespace-nowrap transition-colors ${
                active
                  ? 'bg-red-600 text-white shadow'
                  : 'text-slate-600 dark:text-slate-300 hover:bg-slate-100 dark:hover:bg-slate-800'
              }`}
            >
              {l.label}
              {l.key === 'alerts' && alertCount > 0 && (
                <span className="ml-1 inline-flex items-center justify-center min-w-[18px] h-[18px] px-1 rounded-full bg-red-600 text-white text-[10px] font-bold">
                  {alertCount > 99 ? '99+' : alertCount}
                </span>
              )}
            </button>
          );
        })}
        <span className="ml-auto shrink-0">
          <DarkModeToggle />
        </span>
      </div>
    </nav>
  );
}
