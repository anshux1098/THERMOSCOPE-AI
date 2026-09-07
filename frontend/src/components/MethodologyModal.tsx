import { useEffect } from 'react';

interface MethodologyModalProps {
  isOpen: boolean;
  onClose: () => void;
}

export default function MethodologyModal({ isOpen, onClose }: MethodologyModalProps) {
  useEffect(() => {
    if (!isOpen) return;
    const onKey = (e: KeyboardEvent) => {
      if (e.key === 'Escape') onClose();
    };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, [isOpen, onClose]);

  if (!isOpen) return null;

  return (
    <div
      className="fixed inset-0 bg-slate-900/60 backdrop-blur-sm z-50 flex items-center justify-center p-4"
      onClick={onClose}
      role="dialog"
      aria-modal="true"
      aria-label="Methodology and limitations"
    >
      <div
        className="bg-white dark:bg-slate-900 rounded-2xl shadow-2xl max-w-2xl w-full max-h-[90vh] flex flex-col"
        onClick={(e) => e.stopPropagation()}
      >
        <div className="flex items-center justify-between px-5 py-4 border-b border-slate-200 dark:border-slate-700">
          <h2 className="font-bold text-slate-900 dark:text-slate-100">Methodology &amp; Documented Limitations (SIH Hackathon)</h2>
          <button
            type="button"
            aria-label="Close methodology"
            className="w-7 h-7 flex items-center justify-center rounded-lg hover:bg-slate-100 dark:hover:bg-slate-700 text-slate-500 dark:text-slate-300 text-lg leading-none"
            onClick={onClose}
          >
            ×
          </button>
        </div>

        <div className="overflow-y-auto px-5 py-4 space-y-5 text-sm text-slate-700 dark:text-slate-300">
          <section>
            <h3 className="font-bold text-slate-900 dark:text-slate-100 mb-1.5">1. System Architecture &amp; Hybrid Fusion</h3>
            <p>
              NASA FIRMS hotspots are enriched with OpenStreetMap proximity context (15 km radius),
              scored by 13 rule-based labeling functions, and classified by an XGBoost model.
              A 5-case fusion engine merges rule votes with ML probabilities into a final label,
              calibrated confidence, explanation bullets, and a human-review flag for ambiguous cases.
            </p>
            <svg viewBox="0 0 600 110" width="100%" style={{ maxHeight: 120 }} role="img" aria-label="Pipeline diagram">
              <defs>
                <marker id="arr" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto">
                  <path d="M0,0 L8,4 L0,8 z" fill="#64748b" />
                </marker>
              </defs>
              {[
                { x: 4, label: 'FIRMS' },
                { x: 106, label: 'OSM Context (15km)' },
                { x: 208, label: '13 Labeling Functions' },
                { x: 310, label: 'XGBoost (7-class probs)' },
                { x: 412, label: '5-Case Fusion' },
              ].map((b, i) => (
                <g key={b.label}>
                  <rect x={b.x} y={30} width={94} height={50} rx={8} fill={i === 4 ? '#dbeafe' : '#f1f5f9'} stroke="#64748b" />
                  <text x={b.x + 47} y={52} textAnchor="middle" fontSize={10} fill="#0f172a" fontWeight={700}>
                    {b.label.split(' ')[0]}
                  </text>
                  <text x={b.x + 47} y={66} textAnchor="middle" fontSize={10} fill="#0f172a">
                    {b.label.split(' ').slice(1).join(' ')}
                  </text>
                  {i < 4 && <line x1={b.x + 94} y1={55} x2={b.x + 104} y2={55} stroke="#64748b" strokeWidth={1.5} markerEnd="url(#arr)" />}
                </g>
              ))}
              <g>
                <rect x={510} y={30} width={86} height={50} rx={8} fill="#dcfce7" stroke="#16a34a" />
                <text x={553} y={52} textAnchor="middle" fontSize={10} fill="#0f172a" fontWeight={700}>Output</text>
                <text x={553} y={66} textAnchor="middle" fontSize={9} fill="#0f172a">label + evidence</text>
                <line x1={506} y1={55} x2={509} y2={55} stroke="#64748b" strokeWidth={1.5} markerEnd="url(#arr)" />
              </g>
            </svg>
          </section>

          <section>
            <h3 className="font-bold text-slate-900 dark:text-slate-100 mb-1.5">2. Documented Limitations &amp; Honest Boundaries</h3>
            <ul className="list-disc pl-5 space-y-1">
              <li>Rule-derived training labels inflate accuracy (~99% reported vs ~70–80% honest estimate): the model partly memorizes rule decisions.</li>
              <li>96% of raw hotspots abstain to unclassified — honest abstention under sparse OSM context, routed to human review.</li>
              <li>Only 5 of 7 classes have plentiful samples; mining and process-heat classes are sparse.</li>
              <li>Demo covers 642 hotspots over 3 days across 23 states, with 20,231 industrial + 3,791 forest/agriculture OSM sites.</li>
            </ul>
          </section>

          <section>
            <h3 className="font-bold text-slate-900 dark:text-slate-100 mb-1.5">3. Data Sources</h3>
            <p>
              Thermal detections: NASA FIRMS (VIIRS SNPP/NOAA-20, MODIS). Infrastructure context:
              OpenStreetMap (industrial, refinery, oil/gas, mining, forest, agriculture, power plant layers).
            </p>
          </section>

          <section>
            <h3 className="font-bold text-slate-900 dark:text-slate-100 mb-1.5">4. Intended Use</h3>
            <p>
              Hackathon demonstration only. Automated labels triage analyst attention; every
              low-confidence or conflicting case is flagged for human review before action.
            </p>
            <p className="mt-2 text-xs text-slate-500">
              Dashboard built with React + TypeScript + Tailwind + Leaflet. Data loaded
              from the backend API with a bundled file fallback.
            </p>
          </section>
        </div>

        <div className="px-5 py-3 border-t border-slate-200 flex justify-end">
          <button
            type="button"
            className="text-sm font-semibold bg-red-600 hover:bg-red-700 text-white px-4 py-2 rounded-lg"
            onClick={onClose}
          >
            Got It
          </button>
        </div>
      </div>
    </div>
  );
}
