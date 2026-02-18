const VARIANT_CLASSES: Record<string, string> = {
  // Health / status
  healthy: "bg-success/10 text-success",
  degraded: "bg-warning/10 text-warning",
  unhealthy: "bg-danger/10 text-danger",

  // Model stages
  production: "bg-success/10 text-success",
  staging: "bg-info/10 text-info",
  development: "bg-chart-2/10 text-chart-2",
  retired: "bg-text-muted/10 text-text-muted",

  // Job statuses
  running: "bg-accent/10 text-accent",
  completed: "bg-success/10 text-success",
  failed: "bg-danger/10 text-danger",
  pending: "bg-warning/10 text-warning",
  queued: "bg-text-muted/10 text-text-secondary",
  cancelled: "bg-text-muted/10 text-text-muted",

  // A/B test statuses
  active: "bg-accent/10 text-accent",
  concluded: "bg-success/10 text-success",
  paused: "bg-warning/10 text-warning",

  // Default
  default: "bg-surface-3 text-text-secondary",
};

interface BadgeProps {
  value: string;
  className?: string;
}

export function Badge({ value, className = "" }: BadgeProps) {
  const classes = VARIANT_CLASSES[value.toLowerCase()] ?? VARIANT_CLASSES.default;
  return (
    <span
      className={`inline-flex items-center rounded-full px-2 py-0.5 text-xs font-medium ${classes} ${className}`}
    >
      {value}
    </span>
  );
}
