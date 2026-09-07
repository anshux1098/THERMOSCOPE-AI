interface Props {
  explanation: string[];
}

/**
 * Backend explanation strings sometimes arrive with list-encoding residue
 * (stray double quotes, '", "' separators gluing sentences together).
 * Split glued fragments and strip the residue at render time so every
 * data source (demo + CSV) displays as clean bullets.
 */
export function cleanBullets(raw: string[]): string[] {
  const out: string[] = [];
  for (const b of raw) {
    const parts = String(b ?? '').split(/",\s*"/);
    for (let p of parts) {
      p = p.trim().replace(/^"+|"+$/g, '').replace(/\s+/g, ' ').trim();
      if (!p || /^[.,;:"']+$/.test(p)) continue;
      out.push(p);
    }
  }
  return out;
}

export default function EvidenceSection({ explanation }: Props) {
  const bullets = cleanBullets(explanation);
  return (
    <div>
      <h3 className="flex items-center gap-2 mb-3 text-xs font-semibold uppercase tracking-wider text-slate-500 dark:text-slate-400">
        <svg className="w-3 h-3 text-slate-400" fill="none" stroke="currentColor" viewBox="0 0 24 24" aria-hidden="true">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M8 6h13M8 12h13M8 18h13M3 6h.01M3 12h.01M3 18h.01" />
        </svg>
        Why this classification? (Explainable AI)
      </h3>
      <ul className="list-disc pl-5 space-y-2 text-sm leading-relaxed text-slate-700 dark:text-slate-300">
        {bullets.map((line, i) => (
          <li key={i} className="pl-1">{line}</li>
        ))}
      </ul>
    </div>
  );
}
