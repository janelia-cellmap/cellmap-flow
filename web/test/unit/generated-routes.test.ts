// The route table generated from cellmap_flow/schema.py says which routes the
// browser answers (both, stub, browser) and which it leaves to a server; the
// router must agree with it.
import { describe, expect, it } from "vitest";

import { createRouter } from "../../src/backend";
import { ROUTES } from "../../src/generated/cellmap-flow";

/** Flask's `/job/<job_id>/status` as a request path the router can match. */
const concrete = (path: string) => path.replace(/<[^>]+>/g, "x");

describe("the generated route table", () => {
  const router = createRouter();

  it("has every route the browser answers, and only those", () => {
    expect(ROUTES.length).toBeGreaterThan(0);
    for (const route of ROUTES) {
      const found = router.match(route.method, concrete(route.path));
      const answered = found !== null && "handler" in found;
      expect(answered, `${route.method} ${route.path} is declared ${route.availability}`).toBe(route.availability !== "server");
    }
  });
});
