// Minimal typed access to the global Leaflet (`L`) loaded from CDN.
// Do NOT install the leaflet npm package; index.html provides window.L.

export interface LeafletMap {
  setView(center: [number, number], zoom: number): LeafletMap;
  flyTo(center: [number, number], zoom: number, options?: Record<string, unknown>): void;
  remove(): void;
  on(event: string, handler: (...args: never[]) => void): void;
}

export interface LeafletLayer {
  addTo(map: LeafletMap): LeafletLayer;
  clearLayers(): void;
  getElement?: () => Element | undefined;
  on(event: string, handler: (...args: never[]) => void): LeafletLayer;
  setRadius?(r: number): LeafletLayer;
  setStyle?(style: Record<string, unknown>): LeafletLayer;
  remove?(): void;
}

export function getL(): any {
  const w = window as unknown as { L?: unknown };
  if (!w.L) throw new Error('Leaflet failed to load from CDN (window.L is undefined).');
  return w.L;
}

export function createMap(elementId: string): LeafletMap {
  const L = getL();
  const map = L.map(elementId, { zoomControl: false }).setView([22.0, 78.0], 4.5);
  L.control.zoom({ position: 'bottomright' }).addTo(map);
  L.tileLayer('https://tile.openstreetmap.org/{z}/{x}/{y}.png', {
    maxZoom: 19,
    attribution:
      '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors | NASA FIRMS',
  }).addTo(map);
  return map as LeafletMap;
}

export function createLayerGroup(map: LeafletMap): any {
  const L = getL();
  return L.layerGroup().addTo(map);
}

export function createCircleMarker(lat: number, lon: number, options: Record<string, unknown>): any {
  return getL().circleMarker([lat, lon], options);
}

export function createRippleCircle(lat: number, lon: number, color: string): any {
  return getL().circle([lat, lon], {
    radius: 5,
    color,
    weight: 2,
    opacity: 0.9,
    fillColor: color,
    fillOpacity: 0.6,
    interactive: false,
  });
}
