interface Props {
  clusteringOn: boolean;
  toggleClustering: () => void;
  visibleCount: number;
}

export default function ClusterManager({ clusteringOn, toggleClustering, visibleCount }: Props) {
  return (
    <button
      type="button"
      aria-pressed={clusteringOn}
      title="Cluster nearby markers (auto-expands at zoom 9+)"
      onClick={toggleClustering}
      className={`text-xs font-semibold px-3 py-1.5 rounded-lg border ${
        clusteringOn
          ? 'bg-sky-600 border-sky-600 text-white'
          : 'border-slate-300 dark:border-slate-600 text-slate-700 dark:text-slate-200 hover:bg-slate-100 dark:hover:bg-slate-700'
      }`}
    >
      🔵 Clustered{clusteringOn ? ` (${visibleCount})` : ''}
    </button>
  );
}
