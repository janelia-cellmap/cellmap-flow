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
// This is the one place the browser mode reaches into the dashboard's
// behaviour, and it does so from outside: the dashboard's own files stay as
// they are.

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
