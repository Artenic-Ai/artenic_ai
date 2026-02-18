import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { render, screen, waitFor } from "@testing-library/react";
import { MemoryRouter } from "react-router";
import { describe, expect, it } from "vitest";

import { OverviewPage } from "@/pages/overview";

function renderWithProviders() {
  const queryClient = new QueryClient({
    defaultOptions: {
      queries: { retry: false },
    },
  });

  return render(
    <QueryClientProvider client={queryClient}>
      <MemoryRouter>
        <OverviewPage />
      </MemoryRouter>
    </QueryClientProvider>,
  );
}

describe("OverviewPage", () => {
  it("renders the overview title after loading", async () => {
    renderWithProviders();

    await waitFor(() => {
      expect(screen.getByText("Overview")).toBeInTheDocument();
    });
  });

  it("shows model count", async () => {
    renderWithProviders();

    await waitFor(() => {
      expect(screen.getByText("Models")).toBeInTheDocument();
      expect(screen.getByText("8")).toBeInTheDocument();
    });
  });

  it("shows running jobs count", async () => {
    renderWithProviders();

    await waitFor(() => {
      expect(screen.getByText("Training Jobs")).toBeInTheDocument();
      expect(screen.getByText("3 running")).toBeInTheDocument();
    });
  });

  it("shows health alerts section", async () => {
    renderWithProviders();

    await waitFor(() => {
      expect(screen.getByText("Health Alerts")).toBeInTheDocument();
    });
  });

  it("shows recent activity section", async () => {
    renderWithProviders();

    await waitFor(() => {
      expect(screen.getByText("Recent Activity")).toBeInTheDocument();
    });
  });
});
