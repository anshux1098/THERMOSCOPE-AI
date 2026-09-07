import { RuleEngine } from '../../types/hotspot';

interface Props {
  rule_engine: RuleEngine;
  allLFNAMES: string[];
}

export default function RuleVotesGrid({ rule_engine, allLFNAMES }: Props) {
  const voted = new Map(rule_engine.active_lfs.map((a) => [a.name, a.vote]));

  return (
    <div>
      <h3 className="flex items-center gap-2 mb-3 text-xs font-semibold uppercase tracking-wider text-slate-500 dark:text-slate-400">
        <svg className="w-3 h-3 text-slate-400" fill="none" stroke="currentColor" viewBox="0 0 24 24" aria-hidden="true">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z" />
        </svg>
        Labeling Functions ({allLFNAMES.length} total, {rule_engine.active_votes} active)
      </h3>
      <div className="grid grid-cols-1 sm:grid-cols-2 gap-1.5">
        {allLFNAMES.map((name) => {
          const vote = voted.get(name);
          return (
            <div
              key={name}
              className={`lf-card ${vote ? 'active' : 'abstained'}`}
              title={vote ? `Voted ${vote}` : 'Abstained'}
            >
              <div className="truncate">{name}</div>
              <div className={vote ? 'text-emerald-600 font-medium' : ''}>
                {vote ? `✓ ${vote}` : '- abstained'}
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}
