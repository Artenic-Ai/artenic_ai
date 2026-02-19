import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";

import {
  CardSkeleton,
  DetailSkeleton,
  FadeIn,
  Skeleton,
  StatCardSkeleton,
  TableSkeleton,
} from "@/components/ui/skeleton";

describe("Skeleton", () => {
  it("renders with aria-hidden", () => {
    const { container } = render(<Skeleton className="h-4 w-20" />);
    const el = container.firstChild as HTMLElement;
    expect(el.getAttribute("aria-hidden")).toBe("true");
  });

  it("applies custom className", () => {
    const { container } = render(<Skeleton className="h-8 w-full" />);
    const el = container.firstChild as HTMLElement;
    expect(el.className).toContain("h-8");
    expect(el.className).toContain("w-full");
  });

  it("applies shimmer animation class", () => {
    const { container } = render(<Skeleton />);
    const el = container.firstChild as HTMLElement;
    expect(el.className).toContain("animate-shimmer");
  });
});

describe("StatCardSkeleton", () => {
  it("renders skeleton blocks", () => {
    const { container } = render(<StatCardSkeleton />);
    const skeletons = container.querySelectorAll("[aria-hidden='true']");
    expect(skeletons.length).toBe(3);
  });
});

describe("CardSkeleton", () => {
  it("renders with default height", () => {
    const { container } = render(<CardSkeleton />);
    expect(container.querySelector("[aria-hidden]")).toBeTruthy();
  });

  it("accepts className prop", () => {
    const { container } = render(<CardSkeleton className="lg:col-span-2" />);
    const outer = container.firstChild as HTMLElement;
    expect(outer.className).toContain("lg:col-span-2");
  });
});

describe("TableSkeleton", () => {
  it("renders header plus rows", () => {
    const { container } = render(<TableSkeleton rows={3} cols={2} />);
    const rows = container.querySelectorAll(".border-b");
    expect(rows.length).toBe(4); // 1 header + 3 data
  });
});

describe("DetailSkeleton", () => {
  it("renders two card skeletons", () => {
    const { container } = render(<DetailSkeleton />);
    const cards = container.querySelectorAll(".border-border.bg-surface-1");
    expect(cards.length).toBeGreaterThanOrEqual(2);
  });
});

describe("FadeIn", () => {
  it("wraps children with fade-in class", () => {
    render(
      <FadeIn>
        <span>content</span>
      </FadeIn>,
    );
    expect(screen.getByText("content")).toBeInTheDocument();
    expect(screen.getByText("content").parentElement?.className).toContain(
      "animate-fade-in",
    );
  });
});
