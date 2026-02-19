import {
  Bar,
  BarChart as RechartsBarChart,
  CartesianGrid,
  Legend,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";

interface BarChartSeries {
  key: string;
  label: string;
  color: string;
}

interface BarChartProps {
  data: Array<Record<string, unknown>>;
  xKey: string;
  series: BarChartSeries[];
  height?: number;
  stacked?: boolean;
  formatY?: (value: number) => string;
  xLabel?: string;
  yLabel?: string;
}

export function BarChart({
  data,
  xKey,
  series,
  height = 250,
  stacked = false,
  formatY,
  xLabel,
  yLabel,
}: BarChartProps) {
  return (
    <ResponsiveContainer width="100%" height={height}>
      <RechartsBarChart data={data}>
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
        <Legend
          wrapperStyle={{ color: "var(--color-text-secondary)", fontSize: 12 }}
        />
        {series.map((s) => (
          <Bar
            key={s.key}
            dataKey={s.key}
            name={s.label}
            fill={s.color}
            stackId={stacked ? "stack" : undefined}
            radius={stacked ? undefined : [4, 4, 0, 0]}
          />
        ))}
      </RechartsBarChart>
    </ResponsiveContainer>
  );
}
