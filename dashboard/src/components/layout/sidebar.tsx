import {
  Activity,
  Box,
  Cpu,
  GitBranch,
  Layers,
  LayoutDashboard,
  Settings,
  Wallet,
  Zap,
} from "lucide-react";
import { NavLink } from "react-router";

const NAV_ITEMS = [
  { to: "/", icon: LayoutDashboard, label: "Overview" },
  { to: "/models", icon: Box, label: "Models" },
  { to: "/training", icon: Cpu, label: "Training" },
  { to: "/inference", icon: Zap, label: "Inference" },
  { to: "/ensembles", icon: Layers, label: "Ensembles" },
  { to: "/ab-tests", icon: GitBranch, label: "A/B Tests" },
  { to: "/budgets", icon: Wallet, label: "Budgets" },
  { to: "/health", icon: Activity, label: "Health" },
  { to: "/settings", icon: Settings, label: "Settings" },
] as const;

export function Sidebar() {
  return (
    <aside className="flex h-screen w-56 flex-col border-r border-border bg-surface-1">
      <div className="flex h-14 items-center gap-2 border-b border-border px-4">
        <div className="h-7 w-7 rounded-lg bg-accent" />
        <span className="text-sm font-semibold text-text-primary">
          Artenic AI
        </span>
      </div>

      <nav className="flex-1 space-y-0.5 overflow-y-auto px-2 py-3">
        {NAV_ITEMS.map((item) => (
          <NavLink
            key={item.to}
            to={item.to}
            end={item.to === "/"}
            className={({ isActive }) =>
              `flex items-center gap-3 rounded-md px-3 py-2 text-sm transition-colors ${
                isActive
                  ? "bg-accent/10 text-accent font-medium"
                  : "text-text-secondary hover:bg-surface-2 hover:text-text-primary"
              }`
            }
          >
            <item.icon size={18} />
            {item.label}
          </NavLink>
        ))}
      </nav>

      <div className="border-t border-border px-4 py-3">
        <div className="flex items-center gap-2">
          <span className="inline-block h-2 w-2 rounded-full bg-success" />
          <span className="text-xs text-text-muted">
            {import.meta.env.VITE_DEMO_MODE === "true" ? "Demo" : "Connected"}
          </span>
        </div>
      </div>
    </aside>
  );
}
