import { ALL_LF_NAMES, Hotspot } from '../types/hotspot';
import ConfidenceAssessor from './sections/ConfidenceAssessor';
import ClassificationHeader from './sections/ClassificationHeader';
import MetadataSection from './sections/MetadataSection';
import DataQualityBadge from './DataQualityBadge';
import SpatialContextDiagram from './sections/SpatialContextDiagram';
import SpatialContext from './sections/SpatialContext';
import EvidenceSection from './sections/EvidenceSection';
import HumanAnalystView from './sections/HumanAnalystView';
import RuleVotesGrid from './sections/RuleVotesGrid';
import MlProbabilitiesChart from './sections/MlProbabilitiesChart';

interface DetailPanelProps {
  hotspots: Hotspot[];
  selectedId: number | null;
  onClose: () => void;
  onCopy: (text: string) => void;
}

const DELAYS = [0, 60, 120, 180, 240, 300, 360, 420, 480, 540];

export default function DetailPanel({ hotspots, selectedId, onClose, onCopy }: DetailPanelProps) {
  const open = selectedId !== null;
  const hotspot = open ? hotspots[selectedId as number] : undefined;

  const sections = hotspot
    ? [
        <ConfidenceAssessor key="ca" hotspot={hotspot} />,
        <ClassificationHeader key="ch" classification={hotspot.classification} />,
        <MetadataSection key="md" hotspot={hotspot} onCopy={onCopy} />,
        <DataQualityBadge key="dq" hotspot={hotspot} />,
        <SpatialContextDiagram key="sd" spatial_context={hotspot.spatial_context} />,
        <SpatialContext key="sc" spatial_context={hotspot.spatial_context} />,
        <EvidenceSection key="ev" explanation={hotspot.classification.explanation} />,
        <HumanAnalystView key="ha" hotspot={hotspot} />,
        <RuleVotesGrid key="rv" rule_engine={hotspot.rule_engine} allLFNAMES={ALL_LF_NAMES} />,
        <MlProbabilitiesChart key="ml" ml_engine={hotspot.ml_engine} />,
      ]
    : [];

  return (
    <aside
      className={`detail-panel absolute right-0 top-0 bottom-0 w-full sm:w-[400px] bg-white dark:bg-slate-800 border-l border-slate-200 dark:border-slate-700 shadow-[0_2px_12px_-4px_rgba(0,0,0,0.08)] dark:shadow-[0_2px_12px_-4px_rgba(0,0,0,0.4)] z-[15] flex flex-col ${
        open
          ? 'panel-enter translate-x-0 opacity-100'
          : 'translate-x-full opacity-0 scale-95 pointer-events-none'
      }`}
      aria-hidden={!open}
    >
      <div className="flex items-center justify-between px-4 py-3 border-b border-slate-100 dark:border-slate-700">
        <div>
          <h2 className="font-bold text-slate-900 dark:text-slate-100 text-sm">Hotspot Intelligence Inspector</h2>
          <p className="text-[10px] text-slate-400 dark:text-slate-500">← → Navigate · ESC Close</p>
        </div>
        <button
          type="button"
          aria-label="Close panel"
          className="w-7 h-7 flex items-center justify-center rounded-lg hover:bg-slate-100 dark:hover:bg-slate-700 text-slate-500 dark:text-slate-300 text-lg leading-none transition-colors duration-150 ease-in-out focus-visible:ring-2 focus-visible:ring-offset-2 focus-visible:ring-sky-400"
          onClick={onClose}
        >
          ×
        </button>
      </div>

      {!hotspot ? (
        <div className="flex-1 flex flex-col items-center justify-center text-center px-8">
          <div className="border border-slate-200 dark:border-slate-700 rounded-xl p-6 bg-slate-50/50 dark:bg-slate-800/50">
            <svg viewBox="0 0 24 24" className="w-12 h-12 mx-auto mb-3 text-slate-300" fill="currentColor" aria-hidden="true">
              <path d="M12 2C8.1 2 5 5.1 5 9c0 5.2 7 13 7 13s7-7.8 7-13c0-3.9-3.1-7-7-7zm0 9.5A2.5 2.5 0 1 1 12 6a2.5 2.5 0 0 1 0 5.5z" />
            </svg>
            <p className="text-sm text-slate-500 dark:text-slate-400">
              Click any hotspot marker on the map to inspect classification evidence and spatial context.
            </p>
          </div>
        </div>
      ) : (
        <div className="flex-1 overflow-y-auto bg-slate-50/60 dark:bg-slate-900/40" key={hotspot.id}>
          <div className="p-4 space-y-4">
            {sections.map((s, i) => (
              <div
                key={(s as { key: string }).key ?? i}
                className="content-fade bg-white dark:bg-slate-800 rounded-xl border border-slate-200 dark:border-slate-700 shadow-[0_1px_3px_-1px_rgba(0,0,0,0.05)] p-4"
                style={{ animationDelay: `${DELAYS[Math.min(i, DELAYS.length - 1)]}ms` }}
              >
                {s}
              </div>
            ))}
          </div>
        </div>
      )}
    </aside>
  );
}
