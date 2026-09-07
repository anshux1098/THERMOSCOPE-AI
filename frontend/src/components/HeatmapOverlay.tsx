interface Props {
  visible: boolean;
  intensity: number;
  setIntensity: (v: number) => void;
}

const STOPS = ['#ff0000', '#ff8800', '#ffdd00', '#88ff00', '#00ff00'];

export default function HeatmapOverlay({ visible, intensity, setIntensity }: Props) {
  if (!visible) return null;
  return (
    <div className="absolute left-3 bottom-14 z-[13] w-60 bg-white/95 dark:bg-slate-800/95 backdrop-blur-sm border border-slate-200 dark:border-slate-600 rounded-2xl shadow-xl p-3">
      <div className="flex items-center justify-between mb-1.5">
        <h3 className="text-xs font-bold text-slate-800 dark:text-slate-100">🔥 Heatmap (FRP)</h3>
        <span className="text-[11px] text-slate-500 dark:text-slate-400">×{intensity.toFixed(1)}</span>
      </div>
      <input
        type="range"
        aria-label="Heatmap intensity"
        min={0.1}
        max={2}
        step={0.1}
        value={intensity}
        onChange={(e) => setIntensity(parseFloat(e.target.value))}
        className="w-full accent-red-600"
      />
      <div className="mt-2">
        <div
          className="h-2 rounded-full"
          style={{ background: `linear-gradient(to right, ${STOPS.join(',')})` }}
        />
        <div className="flex justify-between text-[10px] text-slate-500 dark:text-slate-400 mt-0.5">
          <span>low FRP</span>
          <span>high FRP</span>
        </div>
      </div>
    </div>
  );
}
