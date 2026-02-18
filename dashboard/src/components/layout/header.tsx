import { useLocation } from "react-router";

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

  const basePath = "/" + (location.pathname.split("/")[1] ?? "");
  const title = ROUTE_TITLES[basePath] ?? "Artenic AI";

  return (
    <header className="flex h-14 items-center border-b border-border bg-surface-1 px-6">
      <h1 className="text-lg font-semibold text-text-primary">{title}</h1>
    </header>
  );
}
