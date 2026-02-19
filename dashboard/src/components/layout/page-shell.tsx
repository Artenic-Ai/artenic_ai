import type { ReactNode } from "react";

interface PageShellProps {
  title: string;
  description?: string;
  actions?: ReactNode;
  breadcrumb?: ReactNode;
  children: ReactNode;
}

export function PageShell({
  title,
  description,
  actions,
  breadcrumb,
  children,
}: PageShellProps) {
  return (
    <div className="animate-fade-in space-y-6">
      {breadcrumb && <div className="-mb-2">{breadcrumb}</div>}
      <div className="flex items-start justify-between">
        <div>
          <h2 className="text-2xl font-bold text-text-primary">{title}</h2>
          {description && (
            <p className="mt-1 text-sm text-text-secondary">{description}</p>
          )}
        </div>
        {actions && <div className="flex gap-2">{actions}</div>}
      </div>
      {children}
    </div>
  );
}
