import { Hotspot } from '../types/hotspot';
import { parseDate } from '../services/dateUtils';

/** Keep hotspots whose acq_date falls inside [start, end] (inclusive, day precision). */
export function filterByDateRange(
  hotspots: Hotspot[],
  start: string | null,
  end: string | null,
): Hotspot[] {
  if (!start && !end) return hotspots;
  return hotspots.filter((h) => {
    const d = parseDate(h.acq_date);
    if (!d) return true;
    const day = d.toISOString().split('T')[0];
    if (start && day < start) return false;
    if (end && day > end) return false;
    return true;
  });
}
