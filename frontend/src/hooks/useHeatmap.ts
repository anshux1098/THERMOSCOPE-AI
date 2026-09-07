import { useMemo, useState } from 'react';
import { Hotspot } from '../types/hotspot';
import { buildHeatPoints, HeatPoint } from '../lib/leaflet_layers';

export function useHeatmap(hotspots: Hotspot[]) {
  const [heatmapOn, setHeatmapOn] = useState(false);
  const [intensity, setIntensity] = useState(1.0);

  const points: HeatPoint[] = useMemo(
    () => buildHeatPoints(hotspots, intensity),
    [hotspots, intensity],
  );

  return {
    heatmapOn,
    toggleHeatmap: () => setHeatmapOn((v) => !v),
    setHeatmapOn,
    intensity,
    setIntensity: (v: number) => setIntensity(Math.min(2, Math.max(0.1, v))),
    points,
  };
}
