interface Props {
  message: string | null;
  onDismiss: () => void;
}

export default function ErrorToast({ message, onDismiss }: Props) {
  if (!message) return null;
  return (
    <div className="absolute top-2 left-1/2 -translate-x-1/2 z-[16] max-w-[90%]">
      <div
        role="alert"
        className="toast-enter flex items-start gap-2 bg-red-900 text-white text-xs font-medium px-4 py-2.5 rounded-xl shadow-[0_4px_12px_rgba(0,0,0,0.2)]"
      >
        <span aria-hidden="true">⚠</span>
        <span className="flex-1">Error: {message}</span>
        <button
          type="button"
          aria-label="Dismiss error"
          className="font-bold hover:opacity-80 leading-none rounded transition-opacity duration-150 ease-in-out focus-visible:ring-2 focus-visible:ring-offset-2 focus-visible:ring-white"
          onClick={onDismiss}
        >
          ×
        </button>
      </div>
    </div>
  );
}
