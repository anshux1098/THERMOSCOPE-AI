import { useEffect, useRef } from 'react';
import { ClassKey, CLASS_COLORS, Hotspot } from '../types/hotspot';
import {
  createCircleMarker,
  createLayerGroup,
  createMap,
  createRippleCircle,
} from '../lib/leaflet';
import { createClusterGroup, createHeatLayer, createPlacePin, HeatPoint } from '../lib/leaflet_layers';

export interface MapApi {
  flyToIndex: (index: number) => void;
  flyToCoords: (lat: number, lon: number, zoom?: number) => void;
  fitBoundsVisible: () => void;
  addPlacePin: (lat: number, lon: number, label: string) => void;
  clearPlacePins: () => void;
}

export interface PlacePin {
  lat: number;
  lon: number;
  label: string;
}

interface MapProps {
  hotspots: Hotspot[];
  visibleIndices: number[];
  selectedId: number | null;
  selectedIds: Set<number>;
  clusteringOn: boolean;
  heatmapOn: boolean;
  heatPoints: HeatPoint[];
  pins: PlacePin[];
  onMarkerClick: (index: number) => void;
  onPlacePin: (lat: number, lon: number, label: string) => void;
  onClearPins: () => void;
  registerMapApi: (api: MapApi) => void;
}

function esc(s: string): string {
  return s.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
}

function rippleAtMap(m: any, lat: number, lon: number, c: string) {
  try {
    const ripple = createRippleCircle(lat, lon, c);
    ripple.addTo(m);
    const el = ripple.getElement?.();
    if (el) el.classList.add('ripple-ring');
    window.setTimeout(() => {
      try {
        m.removeLayer(ripple);
      } catch {
        /* gone */
      }
    }, 500);
  } catch {
    /* decorative */
  }
}

export default function Map(props: MapProps) {
  const {
    hotspots,
    visibleIndices,
    selectedId,
    selectedIds,
    clusteringOn,
    heatmapOn,
    heatPoints,
    pins,
    onMarkerClick,
    onPlacePin,
    onClearPins,
    registerMapApi,
  } = props;

  const mapRef = useRef<any>(null);
  const layerRef = useRef<any>(null);
  const clusterRef = useRef<any>(null);
  const heatRef = useRef<any>(null);
  const pinsRef = useRef<any>(null);
  const markersRef = useRef<globalThis.Map<number, any>>(new globalThis.Map());
  const prevSelectedRef = useRef<number | null>(null);
  const clickRef = useRef(onMarkerClick);
  clickRef.current = onMarkerClick;
  const dataRef = useRef(hotspots);
  dataRef.current = hotspots;
  const visibleRef = useRef(visibleIndices);
  visibleRef.current = visibleIndices;
  const batchRef = useRef(selectedIds);
  batchRef.current = selectedIds;

  const styleFor = (selected: boolean, inBatch: boolean) =>
    selected
      ? { radius: 14, fillOpacity: 1.0, color: '#1e293b', weight: 3 }
      : inBatch
        ? { radius: 11, fillOpacity: 0.8, color: '#0ea5e9', weight: 3 }
        : { radius: 11, fillOpacity: 0.8, color: '#ffffff', weight: 2 };

  // Init once
  useEffect(() => {
    try {
      const map = createMap('map');
      mapRef.current = map;
      layerRef.current = createLayerGroup(map);
      pinsRef.current = createLayerGroup(map);
    } catch (err) {
      // eslint-disable-next-line no-console
      console.error(err);
    }
    return () => {
      try {
        if (clusterRef.current) mapRef.current?.removeLayer(clusterRef.current);
      } catch {
        /* gone */
      }
      try {
        mapRef.current?.remove();
      } catch {
        /* already removed */
      }
      mapRef.current = null;
    };
  }, []);

  // Map API exposed to App
  useEffect(() => {
    registerMapApi({
      flyToIndex: (index: number) => {
        const map = mapRef.current;
        const h = dataRef.current[index];
        if (!map || !h) return;
        map.flyTo([h.lat, h.lon], 11, { duration: 1.5 });
        window.setTimeout(() => {
          rippleAtMap(map, h.lat, h.lon, CLASS_COLORS[h.classification.class as ClassKey] ?? '#9e9e9e');
          clickRef.current(index);
        }, 1550);
      },
      flyToCoords: (lat: number, lon: number, zoom = 12) => {
        mapRef.current?.flyTo([lat, lon], zoom, { duration: 1.5 });
      },
      fitBoundsVisible: () => {
        const map = mapRef.current;
        const list = visibleRef.current;
        if (!map || list.length === 0) return;
        try {
          const L = (window as unknown as { L: any }).L;
          const bounds = L.latLngBounds(list.map((i) => [dataRef.current[i].lat, dataRef.current[i].lon]));
          map.fitBounds(bounds.pad(0.15));
        } catch {
          /* fit failed */
        }
      },
      addPlacePin: (lat: number, lon: number, label: string) => onPlacePin(lat, lon, label),
      clearPlacePins: () => onClearPins(),
    });
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const buildMarker = (map: any, h: Hotspot, index: number, order: number) => {
    const color = CLASS_COLORS[h.classification.class as ClassKey] ?? '#9e9e9e';
    const selected = index === selectedId;
    const inBatch = selectedIds.has(index);
    const base = styleFor(selected, inBatch);
    const marker = createCircleMarker(h.lat, h.lon, {
      radius: base.radius,
      fillColor: color,
      fillOpacity: base.fillOpacity,
      color: base.color,
      weight: base.weight,
    });
    if (selected) {
      try {
        marker.getElement?.()?.classList.add('marker-selected');
      } catch {
        /* element not ready */
      }
    }
    (marker as any)._hsIndex = order;
    (marker as any)._hsSelected = selected;
    try {
      (marker.options as any).hotspotData = h;
      (marker.options as any).hotspotIndex = index;
    } catch {
      /* options sealed */
    }
    const confPct = (h.classification.confidence * 100).toFixed(0);
    marker.bindTooltip(
      `<div class="custom-tooltip">` +
        `<div class="tooltip-header"><span class="tooltip-dot" style="background:${color}"></span><strong>${esc(h.classification.display_name)}</strong></div>` +
        `<div class="tooltip-body"><div>${h.lat.toFixed(3)}, ${h.lon.toFixed(3)}</div>` +
        `<div>FRP: ${h.frp} MW</div>` +
        `<div class="tooltip-conf-bar"><div style="width:${confPct}%;background:${color};height:100%;border-radius:2px"></div></div>` +
        `<div>Confidence: ${confPct}%</div></div>` +
        `<div class="tooltip-footer">Click to inspect</div></div>`,
      { direction: 'top', offset: [0, -5], className: 'custom-tooltip-container' },
    );
    marker.on('mouseover', function (this: any) {
      this.setRadius(14);
      this.setStyle({ fillOpacity: 1.0 });
    });
    marker.on('mouseout', function (this: any) {
      const sel = (this._hsSelected as boolean) ?? false;
      const batched = (this._hsBatch as boolean) ?? false;
      const s = styleFor(sel, batched);
      this.setRadius(s.radius);
      this.setStyle({ fillOpacity: s.fillOpacity, color: s.color, weight: s.weight });
    });
    marker.on('click', function (this: any) {
      try {
        this.setRadius(9);
        const self = this;
        const sel = (self._hsSelected as boolean) ?? false;
        window.setTimeout(() => {
          try {
            self.setRadius(sel ? 14 : 11);
          } catch {
            /* gone */
          }
        }, 100);
      } catch {
        /* press effect n/a */
      }
      rippleAtMap(map, h.lat, h.lon, color);
      clickRef.current(index);
    });
    return marker;
  };

  const applyEntrance = (marker: any, h: Hotspot, order: number, total: number) => {
    try {
      const el = marker.getElement?.() as Element | undefined;
      if (!el) return;
      // Adaptive: stagger small sets, fade large sets together (1268 x 12ms would take 15s).
      const delay = total <= 50 ? order * 12 : 0;
      (el as HTMLElement).style.animation = `marker-appear 0.3s ease-out ${delay}ms backwards`;
      if (h.frp >= 15) el.classList.add('thermal-pulse');
    } catch {
      /* element not ready */
    }
  };

  // Targeted selected-ring highlight (no full marker rebuild on selection change)
  useEffect(() => {
    const markers = markersRef.current;
    const prev = prevSelectedRef.current;
    if (prev !== null && prev !== selectedId) {
      const m = markers.get(prev);
      if (m) {
        try {
          const inBatch = batchRef.current.has(prev);
          const s = styleFor(false, inBatch);
          m.setRadius(s.radius);
          m.setStyle({ fillOpacity: s.fillOpacity, color: s.color, weight: s.weight });
          m.getElement?.()?.classList.remove('marker-selected');
          (m as any)._hsSelected = false;
        } catch {
          /* gone */
        }
      }
    }
    if (selectedId !== null) {
      const m = markers.get(selectedId);
      if (m) {
        try {
          const inBatch = batchRef.current.has(selectedId);
          const s = styleFor(true, inBatch);
          m.setRadius(s.radius);
          m.setStyle({ fillOpacity: s.fillOpacity, color: s.color, weight: s.weight });
          m.getElement?.()?.classList.add('marker-selected');
          (m as any)._hsSelected = true;
        } catch {
          /* gone */
        }
      }
    }
    prevSelectedRef.current = selectedId;
  }, [selectedId]);

  // Render markers (plain or clustered)
  useEffect(() => {
    const map = mapRef.current;
    const layer = layerRef.current;
    if (!map || !layer) return;

    // Tear down previous containers
    try {
      if (clusterRef.current) {
        map.removeLayer(clusterRef.current);
        clusterRef.current = null;
      }
    } catch {
      /* gone */
    }
    layer.clearLayers();
    markersRef.current.clear();
    prevSelectedRef.current = null;

    const track = (marker: any, index: number) => {
      (marker as any)._hsBatch = selectedIds.has(index);
      markersRef.current.set(index, marker);
    };

    const total = visibleIndices.length;
    const doEntrance = (marker: any, h: Hotspot, order: number) => {
      if (typeof requestAnimationFrame !== 'undefined') {
        requestAnimationFrame(() => applyEntrance(marker, h, order, total));
      } else window.setTimeout(() => applyEntrance(marker, h, order, total), 0);
    };

    if (clusteringOn) {
      const group = createClusterGroup((index: number) => clickRef.current(index));
      if (group) {
        visibleIndices.forEach((index, order) => {
          const h = hotspots[index];
          if (!h) return;
          const marker = buildMarker(map, h, index, order);
          track(marker, index);
          group.addLayer(marker);
          doEntrance(marker, h, order);
        });
        group.addTo(map);
        clusterRef.current = group;
        prevSelectedRef.current = selectedId;
        return;
      }
      // CDN cluster plugin missing -> fall through to plain markers
    }

    visibleIndices.forEach((index, order) => {
      const h = hotspots[index];
      if (!h) return;
      const marker = buildMarker(map, h, index, order);
      track(marker, index);
      marker.addTo(layer);
      doEntrance(marker, h, order);
    });
    prevSelectedRef.current = selectedId;
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [hotspots, visibleIndices, selectedIds, clusteringOn]);

  // Heat layer (below markers)
  useEffect(() => {
    const map = mapRef.current;
    if (!map) return;
    try {
      if (heatRef.current) {
        map.removeLayer(heatRef.current);
        heatRef.current = null;
      }
    } catch {
      /* gone */
    }
    if (heatmapOn && heatPoints.length > 0) {
      const layer = createHeatLayer(map, heatPoints);
      heatRef.current = layer;
      try {
        if (layer && typeof layer.bringToBack === 'function') layer.bringToBack();
      } catch {
        /* z-order n/a */
      }
    }
  }, [heatmapOn, heatPoints]);

  // Place pins
  useEffect(() => {
    const map = mapRef.current;
    const layer = pinsRef.current;
    if (!map || !layer) return;
    layer.clearLayers();
    for (const p of pins) {
      try {
        const pin = createPlacePin(map, p.lat, p.lon, p.label);
        if (pin) layer.addLayer(pin);
      } catch {
        /* pin failed */
      }
    }
  }, [pins]);

  return <div id="map" className="absolute inset-0" role="application" aria-label="Hotspot map of India" />;
}
