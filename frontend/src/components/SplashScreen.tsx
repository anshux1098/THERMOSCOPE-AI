import { useEffect, useState } from 'react';

interface Props {
  visible: boolean;
}

export default function SplashScreen({ visible }: Props) {
  const [leaving, setLeaving] = useState(false);
  const [gone, setGone] = useState(false);

  useEffect(() => {
    if (!visible) {
      setLeaving(true);
      const t = window.setTimeout(() => setGone(true), 300);
      return () => window.clearTimeout(t);
    }
    setLeaving(false);
    setGone(false);
    return undefined;
  }, [visible]);

  if (gone) return null;
  return (
    <div
      className={`fixed inset-0 bg-slate-900 z-[60] flex flex-col items-center justify-center gap-4 transition-opacity duration-300 ease-out ${
        leaving ? 'opacity-0 pointer-events-none' : 'opacity-100'
      }`}
    >
      <svg viewBox="0 0 24 24" className="w-16 h-16 text-amber-500" fill="currentColor" aria-hidden="true">
        <path d="M12 2c1 4-4 5.5-4 10a4.5 4.5 0 009 0c0-1.5-.5-2.5-1-3.5-.8.8-1 1.5-1 2.5C14.5 7 13 4 12 2zm0 20a6.5 6.5 0 01-6.5-6.5c0-.5.1-1 .2-1.5C7 15.5 8.5 17 12 17s5-1.5 6.3-3c.1.5.2 1 .2 1.5A6.5 6.5 0 0112 22z" />
      </svg>
      <div className="font-bold text-xl tracking-tight"><span className="text-slate-50">THERMOSCOPE</span> <span className="text-amber-500 italic">AI</span></div>
      <div className="text-slate-400 text-sm mt-1">Industrial Fire Intelligence Platform</div>
      <div className="spinner spinner-amber" role="status" aria-label="Loading" />
    </div>
  );
}
