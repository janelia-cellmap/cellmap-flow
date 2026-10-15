// What the end-to-end tests share. A spec file must not import another
// (Playwright collects a spec's tests on import, and refuses a describe()
// called from another file), so what two of them need lives here.
import type { Page } from "@playwright/test";

/** Whether `url` asks the service worker for a computed chunk, not the
 * array's metadata: `virtual/<page>/<view>/.../<int>/<int>/<int>`. */
export function isChunkUrl(url: string): boolean {
  const path = new URL(url).pathname;
  if (!path.includes("/virtual/")) return false;
  const segments = path.split("/").filter(Boolean);
  const last = segments[segments.length - 1] ?? "";
  if (/\.(json|zattrs|zarray|zgroup)$/.test(last) || last === "info") return false;
  return segments.length >= 3 && segments.slice(-3).every((s) => /^\d+$/.test(s));
}

/** The view a computed chunk's URL belongs to: the segment after /virtual/<page>/. */
export function viewOf(url: string): string {
  const path = new URL(url).pathname;
  return decodeURIComponent(path.split("/virtual/")[1]?.split("/")[1] ?? "");
}

/** A JSON route of the in-page backend, called from the page's own fetch,
 * which the service worker relays to the backend as the dashboard's
 * requests are: GET without a body, POST with one. */
export async function api<T = Record<string, unknown>>(page: Page, path: string, body?: unknown): Promise<T> {
  return page.evaluate(
    async ([path, body]) => {
      const init: RequestInit = body === undefined ? {} : { method: "POST", headers: { "content-type": "application/json" }, body: JSON.stringify(body) };
      return (await fetch(path, init)).json();
    },
    [path, body] as const,
  );
}
