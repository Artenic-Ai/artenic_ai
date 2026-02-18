const DEMO_MODE = import.meta.env.VITE_DEMO_MODE === "true";

export class ApiError extends Error {
  constructor(
    public status: number,
    message: string,
  ) {
    super(message);
    this.name = "ApiError";
  }
}

export async function apiFetch<T>(
  path: string,
  init?: RequestInit,
): Promise<T> {
  if (DEMO_MODE) {
    const { handleDemoRequest } = await import("@/mocks/handlers");
    return handleDemoRequest<T>(path, init);
  }

  const res = await fetch(path, {
    headers: {
      "Content-Type": "application/json",
      ...init?.headers,
    },
    ...init,
  });

  if (!res.ok) {
    const text = await res.text();
    throw new ApiError(res.status, text || `HTTP ${res.status}`);
  }

  return res.json() as Promise<T>;
}
