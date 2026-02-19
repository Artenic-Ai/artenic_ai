import { render, screen } from "@testing-library/react";
import { MemoryRouter } from "react-router";
import { describe, expect, it } from "vitest";

import { Breadcrumb } from "@/components/ui/breadcrumb";

function renderBreadcrumb(items: { label: string; to?: string }[]) {
  return render(
    <MemoryRouter>
      <Breadcrumb items={items} />
    </MemoryRouter>,
  );
}

describe("Breadcrumb", () => {
  it("renders all items", () => {
    renderBreadcrumb([
      { label: "Models", to: "/models" },
      { label: "sentiment-bert-v3" },
    ]);
    expect(screen.getByText("Models")).toBeInTheDocument();
    expect(screen.getByText("sentiment-bert-v3")).toBeInTheDocument();
  });

  it("renders link for non-last items with to", () => {
    renderBreadcrumb([
      { label: "Models", to: "/models" },
      { label: "Detail" },
    ]);
    const link = screen.getByText("Models").closest("a");
    expect(link).toHaveAttribute("href", "/models");
  });

  it("renders last item as plain text", () => {
    renderBreadcrumb([
      { label: "Models", to: "/models" },
      { label: "Detail" },
    ]);
    const el = screen.getByText("Detail");
    expect(el.tagName).toBe("SPAN");
  });

  it("has aria-label for accessibility", () => {
    renderBreadcrumb([{ label: "Home" }]);
    expect(screen.getByLabelText("Breadcrumb")).toBeInTheDocument();
  });
});
