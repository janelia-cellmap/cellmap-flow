// A hint's "i" (cellmap_flow/dashboard/static/js/lib/dom.js titleHint): a
// mouse resting on it is shown the rest of the hint as a tooltip while the
// hint is closed, and nothing once it is open, the rest then in view. Both
// modes run this script.
import { expect, it } from "vitest";

// @ts-expect-error: the dashboard's scripts are plain JavaScript
import { titleHint } from "../../../cellmap_flow/dashboard/static/js/lib/dom.js";

function hintOf(rest: string) {
  const body = { textContent: rest };
  const details = { open: false, querySelector: (selector: string) => (selector === ".more-body" ? body : null) };
  return { details, summary: { title: "", parentElement: details } };
}

it("titles a closed hint's i with the rest of the hint, and drops it once the hint is open", () => {
  const { details, summary } = hintOf("\n    Asked once per provider\n    and dataset.  ");

  titleHint(summary);
  expect(summary.title).toBe("Asked once per provider and dataset.");

  details.open = true;
  titleHint(summary);
  expect(summary.title).toBe("");
});
