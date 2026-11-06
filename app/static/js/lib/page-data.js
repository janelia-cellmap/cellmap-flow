// The data a page's template hands its scripts, as JSON in
// <script type="application/json" id="page-data">{{ ... | tojson }}</script>,
// so no script needs Jinja inside it. Parsed once per id; the same object is
// returned every time. Throws if the page has no such element.
const parsed = new Map();

export function pageData(id = "page-data") {
  if (!parsed.has(id)) {
    const el = document.getElementById(id);
    if (!el) throw new Error(`This page has no #${id} data island.`);
    parsed.set(id, JSON.parse(el.textContent));
  }
  return parsed.get(id);
}

// Whether the page is the browser mode's (web/): rendered by web/build/
// render.py with mode "browser" in its page-data, where an in-page backend
// answers the requests and there is no server, file system or cluster. The
// Flask dashboard's page says "server" (templates/index.html defaults the
// key). The templates hide what only a server can do by class (server-only);
// a script asks this where it must also stop insisting on it, an output path
// for instance, or must not start a poll of a route the browser cannot answer.
export function isBrowserMode() {
  return pageData().mode === "browser";
}
