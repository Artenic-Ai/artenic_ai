import { lazy, Suspense } from "react";
import { Outlet, createBrowserRouter, RouterProvider } from "react-router";

import { Header } from "@/components/layout/header";
import { Sidebar } from "@/components/layout/sidebar";
import { SidebarProvider } from "@/components/layout/sidebar-context";
import { PageSpinner } from "@/components/ui/spinner";

const OverviewPage = lazy(() =>
  import("@/pages/overview").then((m) => ({ default: m.OverviewPage })),
);
const ModelsListPage = lazy(() =>
  import("@/pages/models/models-list").then((m) => ({
    default: m.ModelsListPage,
  })),
);
const ModelDetailPage = lazy(() =>
  import("@/pages/models/model-detail").then((m) => ({
    default: m.ModelDetailPage,
  })),
);
const TrainingListPage = lazy(() =>
  import("@/pages/training/training-list").then((m) => ({
    default: m.TrainingListPage,
  })),
);
const TrainingDetailPage = lazy(() =>
  import("@/pages/training/training-detail").then((m) => ({
    default: m.TrainingDetailPage,
  })),
);
const InferencePlaygroundPage = lazy(() =>
  import("@/pages/inference/inference-playground").then((m) => ({
    default: m.InferencePlaygroundPage,
  })),
);
const EnsemblesListPage = lazy(() =>
  import("@/pages/ensembles/ensembles-list").then((m) => ({
    default: m.EnsemblesListPage,
  })),
);
const EnsembleDetailPage = lazy(() =>
  import("@/pages/ensembles/ensemble-detail").then((m) => ({
    default: m.EnsembleDetailPage,
  })),
);
const ABTestsListPage = lazy(() =>
  import("@/pages/ab-tests/ab-tests-list").then((m) => ({
    default: m.ABTestsListPage,
  })),
);
const ABTestDetailPage = lazy(() =>
  import("@/pages/ab-tests/ab-test-detail").then((m) => ({
    default: m.ABTestDetailPage,
  })),
);
const BudgetsListPage = lazy(() =>
  import("@/pages/budgets/budgets-list").then((m) => ({
    default: m.BudgetsListPage,
  })),
);
const SettingsPage = lazy(() =>
  import("@/pages/settings/settings-page").then((m) => ({
    default: m.SettingsPage,
  })),
);
const HealthPage = lazy(() =>
  import("@/pages/health/health-page").then((m) => ({
    default: m.HealthPage,
  })),
);

function Layout() {
  return (
    <SidebarProvider>
      <div className="flex h-screen overflow-hidden">
        <Sidebar />
        <div className="flex flex-1 flex-col overflow-hidden">
          <Header />
          <main className="flex-1 overflow-y-auto bg-surface-0 p-4 md:p-6">
            <Suspense fallback={<PageSpinner />}>
              <Outlet />
            </Suspense>
          </main>
        </div>
      </div>
    </SidebarProvider>
  );
}

const router = createBrowserRouter([
  {
    path: "/",
    element: <Layout />,
    children: [
      { index: true, element: <OverviewPage /> },
      { path: "models", element: <ModelsListPage /> },
      { path: "models/:modelId", element: <ModelDetailPage /> },
      { path: "training", element: <TrainingListPage /> },
      { path: "training/:jobId", element: <TrainingDetailPage /> },
      { path: "inference", element: <InferencePlaygroundPage /> },
      { path: "ensembles", element: <EnsemblesListPage /> },
      { path: "ensembles/:ensembleId", element: <EnsembleDetailPage /> },
      { path: "ab-tests", element: <ABTestsListPage /> },
      { path: "ab-tests/:testId", element: <ABTestDetailPage /> },
      { path: "budgets", element: <BudgetsListPage /> },
      { path: "settings", element: <SettingsPage /> },
      { path: "health", element: <HealthPage /> },
      {
        path: "*",
        element: (
          <div className="flex h-64 flex-col items-center justify-center text-center">
            <h2 className="text-lg font-medium text-text-primary">
              404 — Page Not Found
            </h2>
            <p className="mt-1 text-sm text-text-muted">
              The page you are looking for does not exist.
            </p>
          </div>
        ),
      },
    ],
  },
]);

export function App() {
  return <RouterProvider router={router} />;
}
