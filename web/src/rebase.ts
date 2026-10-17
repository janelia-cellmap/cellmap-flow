// Rebase the dashboard's root-relative /api/ URLs onto the page's directory.
//
// The dashboard's JavaScript was written for a Flask app at an origin's root:
// it calls fetch("/api/..."), opens EventSource("/api/..."), and the page has
// <a href="/api/export-config">. On GitHub Pages the app lives under
// /cellmap-flow/app/, and a service worker scoped to that directory never sees
// a request for /api/... at the origin's root. So, before the dashboard's
// script loads, fetch, EventSource and those links are made to resolve /api/
// against the page's directory, where the service worker answers. Served from
// an origin's root (development at http://localhost:5173/) nothing changes.
//
// The dashboard's download links (<a href="/api/..." download>) need more than
// that, wherever the page is served: Chrome sends a download link's request
// to the network, past the service worker, so a click on one is fetched and
// saved here instead (installApiDownloads).
//
// This is the one place the browser mode reaches into the dashboard's
// behaviour, and it does so from outside: the dashboard's own files stay as
// they are.
import { message } from "./shared/message";

/** The directory the page is served from, e.g. https://host/cellmap-flow/app/ . */
export function apiBase(): URL {
  return new URL("./", location.href);
}

/** `url` with a leading /api/ resolved against `base`; any other URL as it is. */
export function rebaseApiUrl(url: string, base: URL): string {
  return url.startsWith("/api/") ? new URL(url.slice(1), base).href : url;
}

/** A same-origin URL object or Request URL at /api/... rebased, else null. */
function rebaseAbsolute(href: string, base: URL): string | null {
  const url = new URL(href, location.href);
  if (url.origin !== base.origin || !url.pathname.startsWith("/api/")) return null;
  return new URL(url.pathname.slice(1) + url.search, base).href;
}

export function installApiRebase(base: URL = apiBase()): void {
  if (base.pathname === "/") return;

  const nativeFetch = window.fetch.bind(window);
  window.fetch = ((input: RequestInfo | URL, init?: RequestInit) => {
    if (typeof input === "string") return nativeFetch(rebaseApiUrl(input, base), init);
    if (input instanceof URL) {
      const rebased = rebaseAbsolute(input.href, base);
      return nativeFetch(rebased ?? input, init);
    }
    const rebased = rebaseAbsolute(input.url, base);
    return nativeFetch(rebased ? new Request(rebased, input) : input, init);
  }) as typeof window.fetch;

  const NativeEventSource = window.EventSource;
  window.EventSource = class extends NativeEventSource {
    constructor(url: string | URL, init?: EventSourceInit) {
      super(typeof url === "string" ? rebaseApiUrl(url, base) : (rebaseAbsolute(url.href, base) ?? url), init);
    }
  } as typeof EventSource;

  for (const a of document.querySelectorAll<HTMLAnchorElement>('a[href^="/api/"]')) {
    a.setAttribute("href", rebaseApiUrl(a.getAttribute("href") ?? "", base));
  }
}

// ------------------------------------------------ download links

/** The URL to fetch for a download link's `href`, resolved against `page`:
 * one on this origin under /api/ (root-relative, or already rebased under
 * `base`), rebased onto `base`; null for any other link, a blob: one among
 * them. */
export function apiDownloadUrl(href: string, page: string | URL, base: URL): string | null {
  const url = new URL(href, page);
  if ((url.protocol !== "http:" && url.protocol !== "https:") || url.origin !== base.origin) return null;
  if (url.pathname.startsWith(`${base.pathname}api/`)) return url.href;
  if (url.pathname.startsWith("/api/")) return new URL(url.pathname.slice(1) + url.search, base).href;
  return null;
}

/** The file name a Content-Disposition header gives: `filename*=UTF-8''...`
 * (RFC 6266) if there is one, else `filename="..."` or a bare `filename=...`;
 * null without either. */
export function dispositionFilename(header: string | null): string | null {
  if (!header) return null;
  const extended = /filename\*\s*=\s*[\w-]*'[^']*'([^;]+)/i.exec(header);
  if (extended) {
    try {
      return decodeURIComponent(extended[1].trim());
    } catch {
      // a malformed escape: the plain filename, if any, below
    }
  }
  const plain = /filename\s*=\s*(?:"([^"]*)"|([^;]+))/i.exec(header);
  return (plain?.[1] ?? plain?.[2])?.trim() || null;
}

/** Download links to /api/ saved through fetch: the Export Config button and
 * the Finetune tab's volume (.zarr.zip) and finetuned model (.onnx)
 * downloads. Chrome sends a download link's request (`<a download>`) to the
 * network, not to the service worker that answers /api/: the server (Vite,
 * GitHub Pages) has no such file, and Chrome reports "File wasn't available
 * on site" (Vite's dev server answered its index.html instead). A click on
 * one is taken here: fetched, which the worker does answer, and the reply
 * saved from a blob: URL under the name its Content-Disposition gives. The
 * link is looked up at the click, so the ones the dashboard fills in later
 * (a volume's, a model's) are taken too. */
export function installApiDownloads(base: URL = apiBase()): void {
  document.addEventListener("click", (event) => {
    if (event.defaultPrevented || event.button !== 0) return;
    const link = (event.target as Element | null)?.closest?.<HTMLAnchorElement>("a[download][href]");
    const url = link ? apiDownloadUrl(link.getAttribute("href") ?? "", location.href, base) : null;
    if (!link || !url) return;
    event.preventDefault();
    void saveDownload(url, link.getAttribute("download"));
  });
}

/** Fetch `url` and save the reply as a file; a failure is said in an alert,
 * as the dashboard says a failed request. */
async function saveDownload(url: string, name: string | null): Promise<void> {
  try {
    const response = await fetch(url);
    if (!response.ok) throw new Error(`${response.status}: ${(await response.text()).slice(0, 300)}`);
    const blob = await response.blob();
    const href = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = href;
    a.download = dispositionFilename(response.headers.get("content-disposition")) || name || new URL(url).pathname.split("/").pop() || "download";
    document.body.append(a);
    a.click();
    a.remove();
    // Chrome reads the blob after the click returns: revoked at once, the
    // download fails as the link's own request did.
    setTimeout(() => URL.revokeObjectURL(href), 60_000);
  } catch (e) {
    console.error("[cellmap-flow] download failed:", url, e);
    alert(`The download failed: ${message(e)}`);
  }
}
