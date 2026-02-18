import type { ReactNode } from "react";

interface StatCardProps {
  title: string;
  value: string | number;
  subtitle?: string;
  icon?: ReactNode;
  trend?: { value: number; label: string };
}

export function StatCard({ title, value, subtitle, icon, trend }: StatCardProps) {
  return (
    <div className="rounded-lg border border-border bg-surface-1 p-5">
      <div className="flex items-center justify-between">
        <span className="text-sm text-text-secondary">{title}</span>
        {icon && <span className="text-text-muted">{icon}</span>}
      </div>
      <div className="mt-2 text-2xl font-bold text-text-primary">{value}</div>
      {(subtitle ?? trend) && (
        <div className="mt-1 flex items-center gap-2 text-xs">
          {trend && (
            <span
              className={
                trend.value >= 0 ? "text-success" : "text-danger"
              }
            >
              {trend.value >= 0 ? "+" : ""}
              {trend.value}% {trend.label}
            </span>
          )}
          {subtitle && <span className="text-text-muted">{subtitle}</span>}
        </div>
      )}
    </div>
  );
}

interface CardProps {
  title?: string;
  className?: string;
  children: ReactNode;
}

export function Card({ title, className = "", children }: CardProps) {
  return (
    <div
      className={`rounded-lg border border-border bg-surface-1 ${className}`}
    >
      {title && (
        <div className="border-b border-border px-5 py-3">
          <h3 className="text-sm font-medium text-text-primary">{title}</h3>
        </div>
      )}
      <div className="p-5">{children}</div>
    </div>
  );
}
