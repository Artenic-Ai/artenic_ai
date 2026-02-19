import { ChevronRight } from "lucide-react";
import { Link } from "react-router";

export interface BreadcrumbItem {
  label: string;
  to?: string;
}

interface BreadcrumbProps {
  items: BreadcrumbItem[];
}

export function Breadcrumb({ items }: BreadcrumbProps) {
  return (
    <nav aria-label="Breadcrumb" className="flex items-center gap-1 text-sm">
      {items.map((item, i) => {
        const isLast = i === items.length - 1;
        return (
          <span key={item.label} className="flex items-center gap-1">
            {i > 0 && <ChevronRight size={14} className="text-text-muted" />}
            {item.to && !isLast ? (
              <Link
                to={item.to}
                className="text-text-secondary transition-colors hover:text-text-primary"
              >
                {item.label}
              </Link>
            ) : (
              <span className="font-medium text-text-primary">{item.label}</span>
            )}
          </span>
        );
      })}
    </nav>
  );
}
