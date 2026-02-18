import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";

import { Badge } from "@/components/ui/badge";

describe("Badge", () => {
  it("renders the value text", () => {
    render(<Badge value="healthy" />);
    expect(screen.getByText("healthy")).toBeInTheDocument();
  });

  it("applies success styling for healthy", () => {
    render(<Badge value="healthy" />);
    const el = screen.getByText("healthy");
    expect(el.className).toContain("text-success");
  });

  it("applies danger styling for failed", () => {
    render(<Badge value="failed" />);
    const el = screen.getByText("failed");
    expect(el.className).toContain("text-danger");
  });

  it("applies warning styling for degraded", () => {
    render(<Badge value="degraded" />);
    const el = screen.getByText("degraded");
    expect(el.className).toContain("text-warning");
  });

  it("applies default styling for unknown values", () => {
    render(<Badge value="custom-status" />);
    const el = screen.getByText("custom-status");
    expect(el.className).toContain("text-text-secondary");
  });

  it("applies custom className", () => {
    render(<Badge value="running" className="my-custom" />);
    const el = screen.getByText("running");
    expect(el.className).toContain("my-custom");
  });

  it("handles case-insensitive matching", () => {
    render(<Badge value="Running" />);
    const el = screen.getByText("Running");
    expect(el.className).toContain("text-accent");
  });
});
