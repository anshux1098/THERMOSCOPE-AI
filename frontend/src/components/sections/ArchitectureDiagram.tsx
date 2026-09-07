const STEPS = [
  { title: 'NASA FIRMS', sub: 'thermal detections' },
  { title: 'OSM Context', sub: '15 km radius' },
  { title: '14 Labeling Functions', sub: 'rule votes' },
  { title: 'XGBoost ML', sub: '7-class probs' },
  { title: '5-Case Fusion', sub: 'calibrated decision' },
  { title: 'Output', sub: 'label + evidence', highlight: true },
];

export default function ArchitectureDiagram() {
  return (
    <svg viewBox="0 0 640 120" width="100%" role="img" aria-label="THERMOSCOPE-AI pipeline diagram">
      <defs>
        <marker id="arch-arr" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto">
          <path d="M0,0 L8,4 L0,8 z" fill="#64748b" />
        </marker>
      </defs>
      {STEPS.map((s, i) => {
        const w = 96;
        const gap = 8;
        const x = 4 + i * (w + gap);
        return (
          <g key={s.title}>
            <rect
              x={x}
              y={30}
              width={w}
              height={56}
              rx={8}
              fill={s.highlight ? '#dcfce7' : '#f1f5f9'}
              stroke={s.highlight ? '#16a34a' : '#64748b'}
            />
            <text x={x + w / 2} y={55} textAnchor="middle" fontSize={10} fill="#0f172a" fontWeight={700}>
              {s.title}
            </text>
            <text x={x + w / 2} y={70} textAnchor="middle" fontSize={9} fill="#475569">
              {s.sub}
            </text>
            {i < STEPS.length - 1 && (
              <line x1={x + w} y1={58} x2={x + w + gap - 1} y2={58} stroke="#64748b" strokeWidth={1.5} markerEnd="url(#arch-arr)" />
            )}
          </g>
        );
      })}
    </svg>
  );
}
