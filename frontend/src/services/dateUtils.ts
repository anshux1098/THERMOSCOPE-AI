import { Hotspot } from '../types/hotspot';

export function parseDate(value: string | undefined | null): Date | null {
  if (!value) return null;
  const d = new Date(value);
  return isNaN(d.getTime()) ? null : d;
}

export function formatDate(d: Date): string {
  return d.toISOString().split('T')[0];
}

export function getDateRange(hotspots: Hotspot[]): { min: Date; max: Date } | null {
  const dates = hotspots
    .map((h) => parseDate(h.acq_date))
    .filter((d): d is Date => d !== null);
  if (dates.length === 0) return null;
  const min = new Date(Math.min(...dates.map((d) => d.getTime())));
  const max = new Date(Math.max(...dates.map((d) => d.getTime())));
  return { min, max };
}

export function countByDate(hotspots: Hotspot[]): { date: string; count: number }[] {
  const map = new Map<string, number>();
  for (const h of hotspots) {
    const d = parseDate(h.acq_date);
    if (!d) continue;
    const key = formatDate(d);
    map.set(key, (map.get(key) ?? 0) + 1);
  }
  return [...map.entries()]
    .map(([date, count]) => ({ date, count }))
    .sort((a, b) => (a.date < b.date ? -1 : 1));
}
