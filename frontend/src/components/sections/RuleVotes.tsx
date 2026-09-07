import { useState } from 'react';
import { RuleEngine } from '../../types/hotspot';

interface Props {
  rule_engine: RuleEngine;
  allLFNAMES: string[];
}

export default function RuleVotes({ rule_engine, allLFNAMES }: Props) {
  const [open, setOpen] = useState(false);
  const voted = new Map(rule_engine.active_lfs.map((a) => [a.name, a.vote]));

  return (
    <div>
      <button
        type="button"
        aria-expanded={open}
        onClick={() => setOpen((v) => !v)}
        className="w-full p-3 bg-slate-50 hover:bg-slate-100 flex items-center justify-between text-left font-bold text-slate-800 text-sm rounded-xl border border-slate-200"
      >
        <span>
          Labeling Functions ({allLFNAMES.length} total, {rule_engine.active_votes} active)
        </span>
        <svg
          viewBox="0 0 24 24"
          className={`w-4 h-4 text-slate-500 transition-transform ${open ? 'rotate-180' : ''}`}
          fill="none"
          stroke="currentColor"
          strokeWidth={2}
          aria-hidden="true"
        >
          <path strokeLinecap="round" strokeLinejoin="round" d="M19 9l-7 7-7-7" />
        </svg>
      </button>
      <div
        className={`overflow-hidden transition-all ease-out ${
          open ? 'max-h-48 opacity-100' : 'max-h-0 opacity-0'
        }`}
        style={{ transitionDuration: '0.25s' }}
      >
        <ul className="overflow-y-auto max-h-48 mt-1 border border-slate-200 rounded-xl divide-y divide-slate-100 bg-white">
          {allLFNAMES.map((name) => {
            const vote = voted.get(name);
            return (
              <li key={name} className="flex items-center justify-between gap-2 px-3 py-1.5 text-xs border-b border-slate-100 last:border-b-0">
                <span className="font-mono text-slate-800 font-medium truncate">{name}</span>
                {vote ? (
                  <span className="text-emerald-600 font-semibold whitespace-nowrap">✓ {vote}</span>
                ) : (
                  <span className="text-slate-400 whitespace-nowrap">— (abstained)</span>
                )}
              </li>
            );
          })}
        </ul>
      </div>
    </div>
  );
}
