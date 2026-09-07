import { useEffect } from 'react';

interface NavOptions {
  visibleIndices: number[];
  selectedId: number | null;
  onSelect: (id: number | null) => void;
  onFlyTo: (lat: number, lon: number) => void;
  getCoords: (index: number) => { lat: number; lon: number } | null;
  toggleClassifiedOnly?: () => void;
}

/** Arrow-key navigation across the *visible* (filtered) marker list. */
export function useKeyboardNav(opts: NavOptions) {
  const { visibleIndices, selectedId, onSelect, onFlyTo, getCoords, toggleClassifiedOnly } = opts;

  useEffect(() => {
    const step = (dir: 1 | -1) => {
      if (visibleIndices.length === 0) return;
      const pos = selectedId !== null ? visibleIndices.indexOf(selectedId) : -1;
      const nextPos = pos === -1 ? (dir === 1 ? 0 : visibleIndices.length - 1) : (pos + dir + visibleIndices.length) % visibleIndices.length;
      const nextIndex = visibleIndices[nextPos];
      onSelect(nextIndex);
      const coords = getCoords(nextIndex);
      if (coords) onFlyTo(coords.lat, coords.lon);
    };

    const onKey = (e: KeyboardEvent) => {
      const tag = (e.target as HTMLElement)?.tagName;
      if (tag === 'INPUT' || tag === 'TEXTAREA' || tag === 'SELECT') return;
      if (e.key === 'ArrowDown' || e.key === 'ArrowRight') {
        e.preventDefault();
        step(1);
      } else if (e.key === 'ArrowUp' || e.key === 'ArrowLeft') {
        e.preventDefault();
        step(-1);
      } else if ((e.key === 'd' || e.key === 'D') && toggleClassifiedOnly) {
        toggleClassifiedOnly();
      }
    };
    document.addEventListener('keydown', onKey);
    return () => document.removeEventListener('keydown', onKey);
  }, [visibleIndices, selectedId, onSelect, onFlyTo, getCoords, toggleClassifiedOnly]);
}
