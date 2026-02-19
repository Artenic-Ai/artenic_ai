import { useEffect, useState } from "react";

import { CheckCircle, Info, X, XCircle, AlertTriangle as Warn } from "lucide-react";

type ToastVariant = "success" | "error" | "warning" | "info";

interface Toast {
  id: string;
  message: string;
  variant: ToastVariant;
}

const VARIANT_STYLES: Record<ToastVariant, string> = {
  success: "border-success/30 bg-success/10 text-success",
  error: "border-danger/30 bg-danger/10 text-danger",
  warning: "border-warning/30 bg-warning/10 text-warning",
  info: "border-info/30 bg-info/10 text-info",
};

const VARIANT_ICONS: Record<ToastVariant, typeof CheckCircle> = {
  success: CheckCircle,
  error: XCircle,
  warning: Warn,
  info: Info,
};

let addToast: (message: string, variant?: ToastVariant) => void = () => {};

export function toast(message: string, variant: ToastVariant = "info") {
  addToast(message, variant);
}

export function ToastContainer() {
  const [toasts, setToasts] = useState<Toast[]>([]);

  useEffect(() => {
    addToast = (message: string, variant: ToastVariant = "info") => {
      const id = crypto.randomUUID();
      setToasts((prev) => [...prev, { id, message, variant }]);
      setTimeout(() => {
        setToasts((prev) => prev.filter((t) => t.id !== id));
      }, 4000);
    };
    return () => {
      addToast = () => {};
    };
  }, []);

  if (toasts.length === 0) return null;

  return (
    <div className="fixed right-4 top-4 z-[100] flex flex-col gap-2">
      {toasts.map((t) => {
        const Icon = VARIANT_ICONS[t.variant];
        return (
          <div
            key={t.id}
            role="alert"
            className={`flex items-center gap-2 rounded-lg border px-4 py-3 shadow-lg animate-slide-in-right ${VARIANT_STYLES[t.variant]}`}
          >
            <Icon size={16} />
            <span className="text-sm font-medium">{t.message}</span>
            <button
              onClick={() =>
                setToasts((prev) => prev.filter((x) => x.id !== t.id))
              }
              className="ml-2 opacity-60 hover:opacity-100"
              aria-label="Dismiss notification"
            >
              <X size={14} />
            </button>
          </div>
        );
      })}
    </div>
  );
}
