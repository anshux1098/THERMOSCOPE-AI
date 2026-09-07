import { useRef, useState } from 'react';
import { Hotspot } from '../types/hotspot';

export type SearchStatus = 'idle' | 'searching' | 'done';

interface SearchResult {
  status: SearchStatus;
  run: (query: string) => void;
}

/** Coordinate parse + Nominatim place search (300ms debounce). */
export function useSearch(
  _hotspots: Hotspot[],
  onFlyToCoords: (lat: number, lon: number) => void,
  onSelectNearest: (lat: number, lon: number) => void,
  showToast: (msg: string) => void,
): SearchResult {
  const [status, setStatus] = useState<SearchStatus>('idle');
  const timer = useRef<number | null>(null);

  const run = (query: string) => {
    if (timer.current !== null) window.clearTimeout(timer.current);
    timer.current = window.setTimeout(() => void doSearch(query.trim()), 300);
  };

  const doSearch = async (q: string) => {
    if (!q) return;
    // Coordinate pattern: "lat, lon" or "lat lon"
    const m = q.match(/^(-?\d+(?:\.\d+)?)[,\s]+(-?\d+(?:\.\d+)?)$/);
    if (m) {
      const lat = parseFloat(m[1]);
      const lon = parseFloat(m[2]);
      if (!Number.isFinite(lat) || !Number.isFinite(lon) || Math.abs(lat) > 90 || Math.abs(lon) > 180) {
        showToast('Invalid coordinates. Use format: lat, lon');
        return;
      }
      setStatus('done');
      onFlyToCoords(lat, lon);
      onSelectNearest(lat, lon);
      return;
    }
    // Place name via Nominatim
    setStatus('searching');
    try {
      const url = `https://nominatim.openstreetmap.org/search?format=json&q=${encodeURIComponent(q)}&limit=5`;
      const res = await fetch(url, { headers: { 'User-Agent': 'THERMOSCOPE-AI-Dashboard/1.0' } });
      const data = (await res.json()) as { lat: string; lon: string; display_name: string }[];
      if (data.length > 0) {
        const lat = parseFloat(data[0].lat);
        const lon = parseFloat(data[0].lon);
        onFlyToCoords(lat, lon);
        showToast(`Found: ${data[0].display_name.split(',').slice(0, 2).join(',')}`);
      } else {
        showToast('Place not found. Try a different search term.');
      }
    } catch {
      showToast('Place not found. Try a different search term.');
    } finally {
      setStatus('done');
    }
  };

  return { status, run };
}
