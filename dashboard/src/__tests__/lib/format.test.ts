import { describe, expect, it } from "vitest";

import {
  formatDate,
  formatDateTime,
  formatEUR,
  formatMs,
  formatNumber,
  formatPercent,
  formatRelative,
} from "@/lib/format";

describe("formatDate", () => {
  it("formats ISO string to en-GB date", () => {
    const result = formatDate("2026-02-15T10:30:00Z");
    expect(result).toContain("Feb");
    expect(result).toContain("2026");
  });
});

describe("formatDateTime", () => {
  it("formats ISO string to date + time", () => {
    const result = formatDateTime("2026-02-15T10:30:00Z");
    expect(result).toContain("Feb");
    expect(result).toContain("2026");
  });
});

describe("formatRelative", () => {
  it("returns seconds for very recent", () => {
    const now = new Date();
    const result = formatRelative(now.toISOString());
    expect(result).toMatch(/\d+s ago/);
  });

  it("returns minutes for recent", () => {
    const fiveMin = new Date(Date.now() - 5 * 60 * 1000);
    const result = formatRelative(fiveMin.toISOString());
    expect(result).toMatch(/\d+m ago/);
  });

  it("returns hours", () => {
    const threeHours = new Date(Date.now() - 3 * 60 * 60 * 1000);
    const result = formatRelative(threeHours.toISOString());
    expect(result).toMatch(/\d+h ago/);
  });

  it("returns days", () => {
    const twoDays = new Date(Date.now() - 2 * 24 * 60 * 60 * 1000);
    const result = formatRelative(twoDays.toISOString());
    expect(result).toMatch(/\d+d ago/);
  });
});

describe("formatEUR", () => {
  it("formats to EUR currency", () => {
    const result = formatEUR(1234.56);
    // de-CH locale uses EUR and typical formatting
    expect(result).toContain("1");
    expect(result).toContain("234");
  });

  it("handles zero", () => {
    const result = formatEUR(0);
    expect(result).toContain("0");
  });
});

describe("formatNumber", () => {
  it("formats millions", () => {
    expect(formatNumber(1_500_000)).toBe("1.5M");
  });

  it("formats thousands", () => {
    expect(formatNumber(45_200)).toBe("45.2K");
  });

  it("formats small numbers", () => {
    expect(formatNumber(42.5)).toBe("42.5");
  });
});

describe("formatPercent", () => {
  it("formats decimal as percent", () => {
    expect(formatPercent(0.912)).toBe("91.2%");
  });

  it("handles custom decimals", () => {
    expect(formatPercent(0.5, 0)).toBe("50%");
  });
});

describe("formatMs", () => {
  it("formats sub-ms as microseconds", () => {
    expect(formatMs(0.5)).toBe("500μs");
  });

  it("formats ms", () => {
    expect(formatMs(45)).toBe("45ms");
  });

  it("formats seconds", () => {
    expect(formatMs(1500)).toBe("1.50s");
  });
});
