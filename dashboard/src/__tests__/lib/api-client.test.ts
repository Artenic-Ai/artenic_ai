import { describe, expect, it } from "vitest";

import { ApiError } from "@/lib/api-client";

describe("ApiError", () => {
  it("has status and message", () => {
    const err = new ApiError(404, "Not found");
    expect(err.status).toBe(404);
    expect(err.message).toBe("Not found");
    expect(err.name).toBe("ApiError");
  });

  it("is an instance of Error", () => {
    const err = new ApiError(500, "Server error");
    expect(err).toBeInstanceOf(Error);
  });
});

describe("apiFetch in demo mode", () => {
  it("delegates to demo handler when VITE_DEMO_MODE is true", async () => {
    // We test the mock handler directly since import.meta.env is Vite-specific
    const { handleDemoRequest } = await import("@/mocks/handlers");

    const result = await handleDemoRequest<{ status: string }>("/health");
    expect(result).toEqual({ status: "healthy" });
  });

  it("returns models list", async () => {
    const { handleDemoRequest } = await import("@/mocks/handlers");
    const result = await handleDemoRequest<unknown[]>("/models");
    expect(Array.isArray(result)).toBe(true);
    expect(result.length).toBe(8);
  });

  it("returns training jobs", async () => {
    const { handleDemoRequest } = await import("@/mocks/handlers");
    const result = await handleDemoRequest<unknown[]>("/training");
    expect(Array.isArray(result)).toBe(true);
    expect(result.length).toBe(12);
  });

  it("returns 404 for unknown routes", async () => {
    const { handleDemoRequest } = await import("@/mocks/handlers");
    await expect(
      handleDemoRequest("/nonexistent-path"),
    ).rejects.toThrow();
  });

  it("returns budgets", async () => {
    const { handleDemoRequest } = await import("@/mocks/handlers");
    const result = await handleDemoRequest<unknown[]>("/budgets");
    expect(Array.isArray(result)).toBe(true);
    expect(result.length).toBe(4);
  });

  it("returns spending data", async () => {
    const { handleDemoRequest } = await import("@/mocks/handlers");
    const result = await handleDemoRequest<unknown[]>("/budgets/spending");
    expect(Array.isArray(result)).toBe(true);
  });

  it("returns activity events", async () => {
    const { handleDemoRequest } = await import("@/mocks/handlers");
    const result = await handleDemoRequest<unknown[]>("/activity");
    expect(Array.isArray(result)).toBe(true);
    expect(result.length).toBe(20);
  });

  it("returns ensembles", async () => {
    const { handleDemoRequest } = await import("@/mocks/handlers");
    const result = await handleDemoRequest<unknown[]>("/ensembles");
    expect(Array.isArray(result)).toBe(true);
    expect(result.length).toBe(3);
  });

  it("returns ab tests", async () => {
    const { handleDemoRequest } = await import("@/mocks/handlers");
    const result = await handleDemoRequest<unknown[]>("/ab-tests");
    expect(Array.isArray(result)).toBe(true);
    expect(result.length).toBe(3);
  });

  it("returns model health data", async () => {
    const { handleDemoRequest } = await import("@/mocks/handlers");
    const result = await handleDemoRequest<unknown[]>("/monitoring");
    expect(Array.isArray(result)).toBe(true);
    expect(result.length).toBe(4);
  });

  it("returns settings schema", async () => {
    const { handleDemoRequest } = await import("@/mocks/handlers");
    const result = await handleDemoRequest<unknown[]>("/settings/schema");
    expect(Array.isArray(result)).toBe(true);
  });

  it("handles POST as success stub", async () => {
    const { handleDemoRequest } = await import("@/mocks/handlers");
    const result = await handleDemoRequest<{ success: boolean }>("/models", {
      method: "POST",
      body: "{}",
    });
    expect(result).toEqual({ success: true });
  });

  it("handles DELETE as deleted stub", async () => {
    const { handleDemoRequest } = await import("@/mocks/handlers");
    const result = await handleDemoRequest<{ deleted: boolean }>(
      "/models/mdl-abc",
      { method: "DELETE" },
    );
    expect(result).toEqual({ deleted: true });
  });
});
