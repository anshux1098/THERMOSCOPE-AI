import { CLASS_COLORS, ClassKey, Hotspot } from '../types/hotspot';

function getL(): any {
  const w = window as unknown as { L?: unknown };
  if (!w.L) throw new Error('Leaflet unavailable');
  return w.L;
}

export type HeatPoint = [number, number, number];

/** FRP-normalized heat points for leaflet.heat. */
export function buildHeatPoints(hotspots: Hotspot[], intensityMult = 1): HeatPoint[] {
  const frps = hotspots.map((h) => h.frp).filter((f) => f > 0);
  const max = frps.length > 0 ? Math.max(...frps) : 1;
  return hotspots
    .filter((h) => h.frp > 0)
    .map((h) => [h.lat, h.lon, Math.min(1, (h.frp / max) * intensityMult)] as HeatPoint);
}

export function createHeatLayer(map: any, points: HeatPoint[]): any | null {
  try {
    const L = getL();
    if (typeof L.heatLayer !== 'function') return null;
    const layer = L.heatLayer(points, {
      radius: 20,
      blur: 15,
      maxZoom: 10,
      max: 1.0,
      gradient: { 0.0: '#ff0000', 0.25: '#ff8800', 0.5: '#ffdd00', 0.75: '#88ff00', 1.0: '#00ff00' },
    });
    layer.addTo(map);
    return layer;
  } catch {
    return null;
  }
}

export function createClusterGroup(onMarkerClick: (index: number) => void): any | null {
  try {
    const L = getL();
    if (typeof L.markerClusterGroup !== 'function') return null;
    return L.markerClusterGroup({
      maxClusterRadius: 60,
      spiderfyOnMaxZoom: true,
      showCoverageOnHover: false,
      zoomToBoundsOnClick: true,
      disableClusteringAtZoom: 9,
      iconCreateFunction: (cluster: any) => {
        const markers: any[] = cluster.getAllChildMarkers();
        const counts: Record<string, number> = {};
        for (const m of markers) {
          const hs = m.options?.hotspotData as Hotspot | undefined;
          if (hs) {
            const cls = hs.classification.class;
            counts[cls] = (counts[cls] || 0) + 1;
          }
        }
        let dominant = 'unclassified';
        let best = 0;
        for (const [cls, n] of Object.entries(counts)) {
          if (n > best) {
            dominant = cls;
            best = n;
          }
        }
        const color = CLASS_COLORS[dominant as ClassKey] ?? '#9e9e9e';
        const count = cluster.getChildCount();
        return L.divIcon({
          html: `<div style="background:${color};color:white;border-radius:50%;width:40px;height:40px;display:flex;align-items:center;justify-content:center;font-weight:bold;font-size:14px;border:2px solid white;box-shadow:0 2px 6px rgba(0,0,0,0.3);">${count > 99 ? '99+' : count}</div>`,
          className: 'cluster-icon',
          iconSize: L.point(40, 40),
        });
      },
    }).on('click', (e: any) => {
      // Child markers handle their own clicks; only fall back here when the
      // event target carries hotspot data but no marker handler fired.
      const idx = e?.layer?.options?.hotspotIndex;
      if (typeof idx === 'number' && e?.layer?.getElement === undefined) onMarkerClick(idx);
    });
  } catch {
    return null;
  }
}

/** Teardrop place pin (non-hotspot search result). */
export function createPlacePin(map: any, lat: number, lon: number, label: string): any | null {
  try {
    const L = getL();
    const icon = L.divIcon({
      html: `<div class="place-pin" title="${label.replace(/"/g, '&quot;')}"></div>`,
      className: 'place-pin-container',
      iconSize: [24, 32],
      iconAnchor: [12, 30],
    });
    const marker = L.marker([lat, lon], { icon, interactive: true });
    marker.bindTooltip(label.length > 60 ? `${label.slice(0, 60)}…` : label, {
      direction: 'top',
      offset: [0, -28],
    });
    marker.addTo(map);
    return marker;
  } catch {
    return null;
  }
}
