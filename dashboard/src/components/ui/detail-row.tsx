import type { ReactNode } from "react";

interface DetailRowProps {
  label: string;
  value: ReactNode;
}

export function DetailRow({ label, value }: DetailRowProps) {
  return (
    <div className="flex items-center justify-between">
      <dt className="text-sm text-text-muted">{label}</dt>
      <dd className="text-sm text-text-primary">{value}</dd>
    </div>
  );
}
