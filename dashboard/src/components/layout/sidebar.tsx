import { useEffect } from "react";
import {
  Activity,
  Box,
  Cpu,
  GitBranch,
  Layers,
  LayoutDashboard,
  Settings,
  Wallet,
  X,
  Zap,
} from "lucide-react";
import { NavLink, useLocation } from "react-router";

import { useSidebar } from "./sidebar-context";

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

function SidebarContent({ onNavigate }: { onNavigate?: () => void }) {
  return (
    <>
      <div className="flex h-14 items-center gap-2 border-b border-border px-4">
        <img src="/favicon.svg" alt="Artenic AI" className="h-7 w-7" />
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
            onClick={onNavigate}
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
    </>
  );
}

export function Sidebar() {
  const { isOpen, close } = useSidebar();
  const location = useLocation();

  // Close mobile sidebar on route change
  useEffect(() => {
    close();
  }, [location.pathname, close]);

  // Close on Escape
  useEffect(() => {
    if (!isOpen) return;
    const handler = (e: KeyboardEvent) => {
      if (e.key === "Escape") close();
    };
    document.addEventListener("keydown", handler);
    return () => document.removeEventListener("keydown", handler);
  }, [isOpen, close]);

  return (
    <>
      {/* Desktop sidebar */}
      <aside className="hidden h-screen w-56 flex-col border-r border-border bg-surface-1 md:flex">
        <SidebarContent />
      </aside>

      {/* Mobile overlay */}
      {isOpen && (
        <div className="fixed inset-0 z-40 md:hidden">
          {/* Backdrop */}
          <div
            className="absolute inset-0 bg-black/50 animate-overlay-fade"
            onClick={close}
            aria-hidden="true"
          />
          {/* Panel */}
          <aside className="relative flex h-full w-64 flex-col bg-surface-1 animate-slide-in-left">
            <button
              onClick={close}
              className="absolute right-2 top-3 rounded-md p-1.5 text-text-muted hover:text-text-primary"
              aria-label="Close sidebar"
            >
              <X size={18} />
            </button>
            <SidebarContent onNavigate={close} />
          </aside>
        </div>
      )}
    </>
  );
}
