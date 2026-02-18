const COLOR_MAP: Record<string, string> = {
  healthy: "bg-success",
  degraded: "bg-warning",
  unhealthy: "bg-danger",
  running: "bg-accent",
  completed: "bg-success",
  failed: "bg-danger",
  pending: "bg-warning",
  active: "bg-accent",
  paused: "bg-warning",
  concluded: "bg-success",
  default: "bg-text-muted",
};

interface StatusDotProps {
  status: string;
  pulse?: boolean;
  className?: string;
}

export function StatusDot({
  status,
  pulse = false,
  className = "",
}: StatusDotProps) {
  const color = COLOR_MAP[status.toLowerCase()] ?? COLOR_MAP.default;
  return (
    <span className={`relative inline-flex ${className}`}>
      <span className={`inline-block h-2 w-2 rounded-full ${color}`} />
      {pulse && (
        <span
          className={`absolute inline-flex h-2 w-2 animate-ping rounded-full opacity-75 ${color}`}
        />
      )}
    </span>
  );
}
