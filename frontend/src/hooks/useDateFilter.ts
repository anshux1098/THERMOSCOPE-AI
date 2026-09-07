import { useEffect, useMemo, useState } from 'react';
import { Hotspot } from '../types/hotspot';
import { formatDate, getDateRange, parseDate } from '../services/dateUtils';

/**
 * Date-range filtering over hotspot acq_date values.
 * State initialized lazily from data; resyncs when the dataset changes.
 */
export function useDateFilter(hotspots: Hotspot[]) {
  const dateRange = useMemo(() => getDateRange(hotspots), [hotspots]);

  const [startDate, setStartDate] = useState<string | null>(null);
  const [endDate, setEndDate] = useState<string | null>(null);

  useEffect(() => {
    setStartDate(dateRange ? formatDate(dateRange.min) : null);
    setEndDate(dateRange ? formatDate(dateRange.max) : null);
  }, [hotspots, dateRange]);

  const filtered = useMemo(() => {
    if (!startDate && !endDate) return hotspots;
    return hotspots.filter((h) => {
      const d = parseDate(h.acq_date);
      if (!d) return true;
      const day = formatDate(d);
      if (startDate && day < startDate) return false;
      if (endDate && day > endDate) return false;
      return true;
    });
  }, [hotspots, startDate, endDate]);

  const clearRange = () => {
    setStartDate(dateRange ? formatDate(dateRange.min) : null);
    setEndDate(dateRange ? formatDate(dateRange.max) : null);
  };

  const isActive =
    dateRange !== null &&
    (startDate !== formatDate(dateRange.min) || endDate !== formatDate(dateRange.max));

  return { startDate, endDate, setStartDate, setEndDate, filtered, dateRange, clearRange, isActive };
}
