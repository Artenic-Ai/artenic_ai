import { useEffect } from "react";
import type { ComponentType } from "react";
import {
  Activity,
  Box,
  Cloud,
  Cpu,
  Database,
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

interface NavItem {
  to: string;
  icon: ComponentType<{ size: number }>;
  label: string;
}

interface NavSection {
  title: string;
  items: NavItem[];
  accentClass: string;
  titleClass: string;
}

const NAV_SECTIONS: NavSection[] = [
  {
    title: "General",
    accentClass: "bg-blue-500/[0.05]",
    titleClass: "text-blue-400/70",
    items: [{ to: "/", icon: LayoutDashboard, label: "Overview" }],
  },
  {
    title: "ML Pipeline",
    accentClass: "bg-purple-500/[0.05]",
    titleClass: "text-purple-400/70",
    items: [
      { to: "/models", icon: Box, label: "Models" },
      { to: "/training", icon: Cpu, label: "Training" },
      { to: "/datasets", icon: Database, label: "Datasets" },
      { to: "/inference", icon: Zap, label: "Inference" },
    ],
  },
  {
    title: "Experimentation",
    accentClass: "bg-cyan-500/[0.05]",
    titleClass: "text-cyan-400/70",
    items: [
      { to: "/ensembles", icon: Layers, label: "Ensembles" },
      { to: "/ab-tests", icon: GitBranch, label: "A/B Tests" },
    ],
  },
  {
    title: "Infrastructure",
    accentClass: "bg-amber-500/[0.05]",
    titleClass: "text-amber-400/70",
    items: [
      { to: "/providers", icon: Cloud, label: "Providers" },
      { to: "/budgets", icon: Wallet, label: "Budgets" },
      { to: "/health", icon: Activity, label: "Health" },
    ],
  },
  {
    title: "System",
    accentClass: "bg-emerald-500/[0.05]",
    titleClass: "text-emerald-400/70",
    items: [{ to: "/settings", icon: Settings, label: "Settings" }],
  },
];

function SidebarContent({ onNavigate }: { onNavigate?: () => void }) {
  return (
    <>
      <div className="flex h-14 items-center gap-2 border-b border-border px-4">
        <img src="/favicon.svg" alt="Artenic AI" className="h-7 w-7" />
        <span className="text-sm font-semibold text-text-primary">
          Artenic AI
        </span>
      </div>

      <nav className="flex-1 space-y-2 overflow-y-auto px-2 py-3">
        {NAV_SECTIONS.map((section) => (
          <div
            key={section.title}
            className={`rounded-lg p-1.5 ${section.accentClass}`}
          >
            <p
              className={`px-2 pt-1 pb-2 text-[10px] font-semibold uppercase tracking-widest ${section.titleClass}`}
            >
              {section.title}
            </p>
            <div className="space-y-0.5">
              {section.items.map((item) => (
                <NavLink
                  key={item.to}
                  to={item.to}
                  end={item.to === "/"}
                  onClick={onNavigate}
                  className={({ isActive }) =>
                    `flex items-center gap-3 rounded-md px-3 py-2 text-[13px] transition-colors ${
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
            </div>
          </div>
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
