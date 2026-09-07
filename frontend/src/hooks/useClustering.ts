import { useState } from 'react';

export function useClustering() {
  const [clusteringOn, setClusteringOn] = useState(false);
  return {
    clusteringOn,
    toggleClustering: () => setClusteringOn((v) => !v),
    setClusteringOn,
  };
}
