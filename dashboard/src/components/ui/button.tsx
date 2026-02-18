import type { ButtonHTMLAttributes, ReactNode } from "react";

type Variant = "primary" | "secondary" | "destructive" | "ghost";

const VARIANT_CLASSES: Record<Variant, string> = {
  primary:
    "bg-accent text-white hover:bg-accent-hover focus-visible:ring-accent/50",
  secondary:
    "bg-surface-3 text-text-primary hover:bg-border-hover focus-visible:ring-border-hover/50",
  destructive:
    "bg-danger/10 text-danger hover:bg-danger/20 focus-visible:ring-danger/50",
  ghost:
    "text-text-secondary hover:bg-surface-2 hover:text-text-primary focus-visible:ring-border-hover/50",
};

interface ButtonProps extends ButtonHTMLAttributes<HTMLButtonElement> {
  variant?: Variant;
  children: ReactNode;
}

export function Button({
  variant = "primary",
  className = "",
  children,
  ...props
}: ButtonProps) {
  return (
    <button
      className={`inline-flex items-center justify-center gap-2 rounded-md px-3 py-1.5 text-sm font-medium transition-colors focus-visible:outline-none focus-visible:ring-2 disabled:pointer-events-none disabled:opacity-50 ${VARIANT_CLASSES[variant]} ${className}`}
      {...props}
    >
      {children}
    </button>
  );
}
