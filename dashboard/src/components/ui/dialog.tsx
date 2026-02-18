import { type ReactNode, useEffect, useRef } from "react";

import { X } from "lucide-react";

interface DialogProps {
  open: boolean;
  onClose: () => void;
  title: string;
  children: ReactNode;
}

export function Dialog({ open, onClose, title, children }: DialogProps) {
  const dialogRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (!open) return;

    function handleKeyDown(e: KeyboardEvent) {
      if (e.key === "Escape") onClose();
    }

    document.addEventListener("keydown", handleKeyDown);
    return () => document.removeEventListener("keydown", handleKeyDown);
  }, [open, onClose]);

  useEffect(() => {
    if (open) {
      dialogRef.current?.focus();
    }
  }, [open]);

  if (!open) return null;

  const titleId = `dialog-title-${title.replace(/\s+/g, "-").toLowerCase()}`;

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center">
      <div
        className="absolute inset-0 bg-black/60"
        onClick={onClose}
        aria-hidden="true"
      />
      <div
        ref={dialogRef}
        role="dialog"
        aria-modal="true"
        aria-labelledby={titleId}
        tabIndex={-1}
        className="relative w-full max-w-lg rounded-lg border border-border bg-surface-1 shadow-xl outline-none"
      >
        <div className="flex items-center justify-between border-b border-border px-5 py-4">
          <h3
            id={titleId}
            className="text-lg font-semibold text-text-primary"
          >
            {title}
          </h3>
          <button
            onClick={onClose}
            aria-label="Close dialog"
            className="rounded-md p-1 text-text-muted hover:bg-surface-3 hover:text-text-primary"
          >
            <X size={18} />
          </button>
        </div>
        <div className="p-5">{children}</div>
      </div>
    </div>
  );
}
