import {
  CartesianGrid,
  Legend,
  Line,
  LineChart as RechartsLineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";

interface LineChartSeries {
  key: string;
  label: string;
  color: string;
}

interface LineChartProps {
  data: Array<Record<string, unknown>>;
  xKey: string;
  series: LineChartSeries[];
  height?: number;
  formatY?: (value: number) => string;
}

export function LineChart({
  data,
  xKey,
  series,
  height = 250,
  formatY,
}: LineChartProps) {
  return (
    <ResponsiveContainer width="100%" height={height}>
      <RechartsLineChart data={data}>
        <CartesianGrid strokeDasharray="3 3" stroke="var(--color-border)" />
        <XAxis
          dataKey={xKey}
          stroke="var(--color-text-muted)"
          fontSize={12}
          tickLine={false}
        />
        <YAxis
          stroke="var(--color-text-muted)"
          fontSize={12}
          tickLine={false}
          tickFormatter={formatY}
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
        {series.length > 1 && (
          <Legend
            wrapperStyle={{
              color: "var(--color-text-secondary)",
              fontSize: 12,
            }}
          />
        )}
        {series.map((s) => (
          <Line
            key={s.key}
            type="monotone"
            dataKey={s.key}
            name={s.label}
            stroke={s.color}
            strokeWidth={2}
            dot={{ r: 3, fill: s.color }}
            activeDot={{ r: 5 }}
          />
        ))}
      </RechartsLineChart>
    </ResponsiveContainer>
  );
}
