// The router: patterns and their parameters, the catch-all, the 405, and the
// answer helpers' shapes.
import { describe, expect, it } from "vitest";

import { badRequest, json, objectBody, Router, text, unavailable } from "../../src/backend/router";
import { fakeContext, request } from "./backend-fakes";

describe("Router.match", () => {
  const router = new Router()
    .get("/api/finetune/jobs", () => json(200, "jobs"))
    .get("/api/finetune/job/:id/status", (_r, p) => json(200, p))
    .post("/api/finetune/job/:id/cancel", (_r, p) => json(200, p))
    .put("/api/pipeline", () => json(200, "put"));

  it("matches literal paths and binds :params, decoding them", () => {
    const m = router.match("GET", "/api/finetune/job/abc%2F1/status");
    expect(m && "params" in m ? m.params : null).toEqual({ id: "abc/1" });
    expect(router.match("GET", "/api/finetune/jobs")).toHaveProperty("handler");
  });

  it("does not match a longer or shorter path, and ignores a trailing slash", () => {
    expect(router.match("GET", "/api/finetune/job/abc")).toBeNull();
    expect(router.match("GET", "/api/finetune/job/abc/status/extra")).toBeNull();
    expect(router.match("GET", "/api/finetune/jobs/")).toHaveProperty("handler");
  });

  it("tells a known path asked with another method apart from an unknown path", () => {
    expect(router.match("DELETE", "/api/finetune/job/abc/status")).toEqual({ methods: ["GET"] });
    expect(router.match("GET", "/api/nothing")).toBeNull();
  });
});

describe("Router.handle", () => {
  it("runs the matched handler with the params and the context", async () => {
    const ctx = fakeContext();
    const router = new Router().get("/api/finetune/job/:id/status", (req, params, c) => json(200, { id: params.id, q: req.query.get("q"), same: c === ctx }));
    const res = await router.handle(request("GET", "/api/finetune/job/j1/status?q=2"), ctx);
    expect(res).toEqual({ status: 200, json: { id: "j1", q: "2", same: true } });
  });

  it("answers unavailable for an unknown route by default, and the catch-all given", async () => {
    const ctx = fakeContext();
    const res = await new Router().handle(request("GET", "/api/review/status"), ctx);
    expect(res).toEqual({ status: 503, json: { success: false, available: false, error: "GET /api/review/status is not a route of the dashboard" } });
    const custom = new Router().otherwise((req) => json(404, { path: req.path }));
    expect(await custom.handle(request("GET", "/api/x"), ctx)).toEqual({ status: 404, json: { path: "/api/x" } });
  });

  it("answers 405 for a known path asked with another method", async () => {
    const router = new Router().post("/api/set-data", () => json(200, {}));
    const res = await router.handle(request("GET", "/api/set-data"), fakeContext());
    expect(res.status).toBe(405);
    expect("json" in res && (res.json as { error: string }).error).toBe("GET is not allowed for /api/set-data (POST)");
  });
});

describe("the answer helpers", () => {
  it("shape their responses", () => {
    expect(json(201, { a: 1 })).toEqual({ status: 201, json: { a: 1 } });
    expect(text(200, "hi")).toEqual({ status: 200, body: "hi", type: "text/plain; charset=utf-8" });
    expect(text(200, "x: 1\n", "text/yaml", { "content-disposition": "attachment" })).toEqual({
      status: 200,
      body: "x: 1\n",
      type: "text/yaml",
      headers: { "content-disposition": "attachment" },
    });
    expect(badRequest("ref is required")).toEqual({ status: 400, json: { success: false, error: "ref is required" } });
    expect(unavailable("no cluster")).toEqual({ status: 503, json: { success: false, available: false, error: "no cluster" } });
  });

  it("objectBody gives the body when it is a JSON object, else null", () => {
    expect(objectBody(request("POST", "/api/x", { a: 1 }))).toEqual({ a: 1 });
    expect(objectBody(request("POST", "/api/x", [1]))).toBeNull();
    expect(objectBody(request("POST", "/api/x", "text"))).toBeNull();
    expect(objectBody(request("POST", "/api/x"))).toBeNull();
  });
});
