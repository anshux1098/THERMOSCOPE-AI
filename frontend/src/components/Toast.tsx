export type ToastKind = 'info' | 'success' | 'warning' | 'error';

interface ToastProps {
  message: string;
  visible: boolean;
  type?: ToastKind;
}

const KIND_BG: Record<ToastKind, string> = {
  info: 'bg-sky-900',
  success: 'bg-emerald-900',
  warning: 'bg-amber-900',
  error: 'bg-red-900',
};

export default function Toast({ message, visible, type = 'info' }: ToastProps) {
  return (
    <div
      role="status"
      aria-live="polite"
      className={`app-toast fixed bottom-16 left-1/2 -translate-x-1/2 ${KIND_BG[type]} text-white text-xs px-4 py-2.5 rounded-xl shadow-[0_4px_12px_rgba(0,0,0,0.2)] z-20 ${
        visible ? 'toast-enter opacity-100 pointer-events-auto' : 'opacity-0 pointer-events-none translate-y-4'
      }`}
    >
      {message}
    </div>
  );
}
