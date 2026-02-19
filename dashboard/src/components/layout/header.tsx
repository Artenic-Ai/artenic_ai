import { Menu } from "lucide-react";
import { useLocation } from "react-router";

import { useSidebar } from "./sidebar-context";

const ROUTE_TITLES: Record<string, string> = {
  "/": "Overview",
  "/models": "Models",
  "/training": "Training",
  "/inference": "Inference",
  "/ensembles": "Ensembles",
  "/ab-tests": "A/B Tests",
  "/budgets": "Budgets",
  "/health": "Health",
  "/settings": "Settings",
};

export function Header() {
  const location = useLocation();
  const { toggle } = useSidebar();

  const basePath = "/" + (location.pathname.split("/")[1] ?? "");
  const title = ROUTE_TITLES[basePath] ?? "Artenic AI";

  return (
    <header className="flex h-14 items-center gap-3 border-b border-border bg-surface-1 px-4 md:px-6">
      <button
        onClick={toggle}
        className="rounded-md p-1.5 text-text-muted hover:text-text-primary md:hidden"
        aria-label="Toggle sidebar"
      >
        <Menu size={20} />
      </button>
      <h1 className="text-lg font-semibold text-text-primary">{title}</h1>
    </header>
  );
}
