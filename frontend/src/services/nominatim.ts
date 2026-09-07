const NOMINATIM_URL = 'https://nominatim.openstreetmap.org/search';
const USER_AGENT = 'THERMOSCOPE-AI-Dashboard/1.0 (SIH 2026)';

export interface NominatimResult {
  place_id: number;
  lon: string;
  lat: string;
  display_name: string;
  type: string;
  class: string;
}

export async function searchPlace(query: string, signal?: AbortSignal): Promise<NominatimResult[]> {
  const url = `${NOMINATIM_URL}?format=json&q=${encodeURIComponent(query)}&limit=5&addressdetails=0`;
  const res = await fetch(url, {
    headers: { 'User-Agent': USER_AGENT },
    signal,
  });
  if (!res.ok) throw new Error('Geocoding service unavailable');
  return (await res.json()) as Promise<NominatimResult[]>;
}

export function calculateRateLimitDelay(lastCallTime: number | null): number {
  const MIN_DELAY = 1000;
  if (lastCallTime === null) return 0;
  const elapsed = Date.now() - lastCallTime;
  return Math.max(0, MIN_DELAY - elapsed);
}
