interface Props {
  visible: boolean;
}

export default function LoadingOverlay({ visible }: Props) {
  if (!visible) return null;
  return (
    <div className="absolute inset-0 bg-slate-900/40 z-20 flex flex-col items-center justify-center gap-3 transition-opacity duration-200 ease-out">
      <div
        className="w-8 h-8 rounded-full border-[3px] border-slate-200 border-t-sky-500 animate-spin"
        role="status"
        aria-label="Loading hotspots"
      />
      <div className="text-sm text-white/80">Loading hotspots...</div>
    </div>
  );
}
