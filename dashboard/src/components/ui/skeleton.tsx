import type { ReactNode } from "react";

interface SkeletonProps {
  className?: string;
}

export function Skeleton({ className = "" }: SkeletonProps) {
  return (
    <div className={`animate-shimmer rounded-md ${className}`} aria-hidden="true" />
  );
}

export function StatCardSkeleton() {
  return (
    <div className="rounded-lg border border-border bg-surface-1 p-5">
      <Skeleton className="h-4 w-24" />
      <Skeleton className="mt-3 h-7 w-16" />
      <Skeleton className="mt-2 h-3 w-20" />
    </div>
  );
}

export function CardSkeleton({
  height = "h-[200px]",
  className = "",
}: {
  height?: string;
  className?: string;
}) {
  return (
    <div className={`rounded-lg border border-border bg-surface-1 ${className}`}>
      <div className="border-b border-border px-5 py-3">
        <Skeleton className="h-4 w-32" />
      </div>
      <div className="p-5">
        <Skeleton className={`w-full ${height}`} />
      </div>
    </div>
  );
}

export function TableSkeleton({
  rows = 5,
  cols = 4,
}: {
  rows?: number;
  cols?: number;
}) {
  return (
    <div className="rounded-lg border border-border bg-surface-1">
      <div className="flex gap-4 border-b border-border px-4 py-3">
        {Array.from({ length: cols }, (_, i) => (
          <Skeleton key={i} className="h-3 flex-1" />
        ))}
      </div>
      {Array.from({ length: rows }, (_, i) => (
        <div key={i} className="flex gap-4 border-b border-border px-4 py-3">
          {Array.from({ length: cols }, (_, j) => (
            <Skeleton key={j} className="h-4 flex-1" />
          ))}
        </div>
      ))}
    </div>
  );
}

export function DetailSkeleton() {
  return (
    <div className="space-y-6 animate-fade-in">
      <div>
        <Skeleton className="h-3 w-32" />
        <Skeleton className="mt-3 h-7 w-48" />
      </div>
      <div className="grid grid-cols-1 gap-6 lg:grid-cols-2">
        <CardSkeleton height="h-[180px]" />
        <CardSkeleton height="h-[180px]" />
      </div>
    </div>
  );
}

export function FadeIn({ children }: { children: ReactNode }) {
  return <div className="animate-fade-in">{children}</div>;
}
