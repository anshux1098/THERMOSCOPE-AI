import { Hotspot } from '../../types/hotspot';

interface Props {
  hotspot: Hotspot;
  onCopy: (text: string) => void;
}

function Row({ label, children }: { label: string; children: React.ReactNode }) {
  return (
    <div className="flex justify-between items-center gap-2 py-2 border-b border-slate-100 dark:border-slate-700 last:border-0 text-xs">
      <span className="text-slate-500 dark:text-slate-400 font-medium">{label}</span>
      <span className="font-mono text-sm font-medium text-slate-800 dark:text-slate-200 text-right">{children}</span>
    </div>
  );
}

function cap(s: string) {
  return s.charAt(0).toUpperCase() + s.slice(1);
}

export default function MetadataSection({ hotspot, onCopy }: Props) {
  const coords = `${hotspot.lat.toFixed(4)}, ${hotspot.lon.toFixed(4)}`;

  const copy = async () => {
    try {
      await navigator.clipboard.writeText(coords);
    } catch {
      const ta = document.createElement('textarea');
      ta.value = coords;
      document.body.appendChild(ta);
      ta.select();
      document.execCommand('copy');
      document.body.removeChild(ta);
    }
    onCopy('Coordinates copied to clipboard!');
  };

  return (
    <div>
      <h3 className="flex items-center gap-2 mb-3 text-xs font-semibold uppercase tracking-wider text-slate-500 dark:text-slate-400">
        <svg className="w-3 h-3 text-slate-400" fill="none" stroke="currentColor" viewBox="0 0 24 24" aria-hidden="true">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M3 3h7v7H3V3M3 17h7v-7H3v7M13 3h7v7h-7V3M13 17h7v-7h-7v7" />
        </svg>
        Hotspot Metadata
      </h3>
      <div>
        <Row label="Hotspot ID">
          <span>{hotspot.id}</span>
        </Row>
        <Row label="Coordinates">
          <span className="inline-flex items-center gap-1.5">
            <span>{coords}</span>
            <button
              type="button"
              aria-label="Copy coordinates"
              className="text-[11px] font-semibold text-red-600 hover:text-red-700 border border-slate-300 dark:border-slate-600 rounded-lg px-1.5 py-0.5 bg-white dark:bg-slate-700 transition-colors duration-150 ease-in-out focus-visible:ring-2 focus-visible:ring-offset-2 focus-visible:ring-sky-400"
              onClick={copy}
            >
              Copy
            </button>
          </span>
        </Row>
        <Row label="FRP">{hotspot.frp} MW</Row>
        <Row label="Brightness">{hotspot.bright_ti4} K</Row>
        <Row label="Detection confidence">{cap(hotspot.confidence)}</Row>
        <Row label="Day / Night">{hotspot.daynight === 'N' ? '🌙 Night' : '☀️ Day'}</Row>
        <Row label="Acquisition date">{hotspot.acq_date}</Row>
        <Row label="Satellite sensor">{hotspot.satellite}</Row>
      </div>
    </div>
  );
}
