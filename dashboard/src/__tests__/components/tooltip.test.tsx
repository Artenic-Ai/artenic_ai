import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";

import { Tooltip } from "@/components/ui/tooltip";

describe("Tooltip", () => {
  it("renders children", () => {
    render(
      <Tooltip content="Help text">
        <button>Hover me</button>
      </Tooltip>,
    );
    expect(screen.getByText("Hover me")).toBeInTheDocument();
  });

  it("renders tooltip content with role=tooltip", () => {
    render(
      <Tooltip content="Help text">
        <span>Target</span>
      </Tooltip>,
    );
    expect(screen.getByRole("tooltip")).toHaveTextContent("Help text");
  });

  it("starts hidden (opacity-0)", () => {
    render(
      <Tooltip content="Info">
        <span>x</span>
      </Tooltip>,
    );
    const tip = screen.getByRole("tooltip");
    expect(tip.className).toContain("opacity-0");
  });

  it("supports position variants", () => {
    render(
      <Tooltip content="Below" position="bottom">
        <span>target</span>
      </Tooltip>,
    );
    const tip = screen.getByRole("tooltip");
    expect(tip.className).toContain("top-full");
  });
});
