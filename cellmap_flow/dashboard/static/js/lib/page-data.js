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
