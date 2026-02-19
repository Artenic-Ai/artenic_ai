import {
  Area,
  AreaChart as RechartsAreaChart,
  CartesianGrid,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";

interface AreaChartProps {
  data: Array<Record<string, unknown>>;
  xKey: string;
  yKey: string;
  color?: string;
  height?: number;
  formatY?: (value: number) => string;
  xLabel?: string;
  yLabel?: string;
}

export function AreaChart({
  data,
  xKey,
  yKey,
  color = "var(--color-chart-1)",
  height = 200,
  formatY,
  xLabel,
  yLabel,
}: AreaChartProps) {
  return (
    <ResponsiveContainer width="100%" height={height}>
      <RechartsAreaChart data={data}>
        <defs>
          <linearGradient id={`grad-${yKey}`} x1="0" y1="0" x2="0" y2="1">
            <stop offset="0%" stopColor={color} stopOpacity={0.3} />
            <stop offset="100%" stopColor={color} stopOpacity={0} />
          </linearGradient>
        </defs>
        <CartesianGrid strokeDasharray="3 3" stroke="var(--color-border)" />
        <XAxis
          dataKey={xKey}
          stroke="var(--color-text-muted)"
          fontSize={12}
          tickLine={false}
          label={xLabel ? { value: xLabel, position: "insideBottom", offset: -5, fill: "var(--color-text-muted)", fontSize: 11 } : undefined}
        />
        <YAxis
          stroke="var(--color-text-muted)"
          fontSize={12}
          tickLine={false}
          tickFormatter={formatY}
          label={yLabel ? { value: yLabel, angle: -90, position: "insideLeft", fill: "var(--color-text-muted)", fontSize: 11 } : undefined}
        />
        <Tooltip
          contentStyle={{
            backgroundColor: "var(--color-surface-2)",
            border: "1px solid var(--color-border)",
            borderRadius: 8,
            color: "var(--color-text-primary)",
            fontSize: 12,
          }}
        />
        <Area
          type="monotone"
          dataKey={yKey}
          stroke={color}
          strokeWidth={2}
          fill={`url(#grad-${yKey})`}
        />
      </RechartsAreaChart>
    </ResponsiveContainer>
  );
}
