// Publish to Hugging Face (backend/routes/publish.ts, src/publish): the
// routes over the backend fakes, with @huggingface/hub itself talking to a
// fake Hub (globalThis.fetch stubbed), so what is checked is what reaches
// Hugging Face: the files one publish commits, the manifest among them read
// back as the catalog reads one, the record that says where the model went,
// the refusals, and the Sign in popup's hand-over to the page. A refusal is
// checked by its status, not its words.
import { afterEach, describe, expect, it, vi } from "vitest";

import { createRouter } from "../../src/backend";
import { SIGN_IN_EVENT } from "../../src/backend/routes/publish";
import type { ApiResponse, ModelEntry } from "../../src/contracts";
import { finetunedRecord } from "../../src/models/finetuned";
import { parseManifest } from "../../src/models/manifest";
import { PUBLIC_APP } from "../../src/publish/model";
import { handOver } from "../../src/publish/oauth";
import { type FakeContext, fakeContext, request } from "./backend-fakes";

function body(res: ApiResponse): any {
  if (!("json" in res)) throw new Error(`not a JSON answer: ${JSON.stringify(res)}`);
  return res.json;
}

const NAME = "mito_aff_setup54_16nm-ft1";
const RAW = "https://example.org/data.zarr";
const ONNX = new Uint8Array([8, 1, 18, 4, 116, 101, 115, 116]);

/** A fake Hugging Face, as far as a publish goes: whoami-v2 for its tokens
 * (each a user of the organization "cellmap"), repos/create, a commit's
 * requests (preupload, the LFS batch answered with a basic transfer, the
 * commit) and the storage's PUT. `committed` holds each commit's files by
 * path; `refuse` answers the repository and commit requests with that
 * status (403: a token that may not write). */
function fakeHub(users: Record<string, string>, refuse: number | null = null) {
  const repos = new Set<string>();
  const stored = new Map<string, Uint8Array>();
  const committed: Map<string, Uint8Array>[] = [];
  const answer = (value: unknown, status = 200) => new Response(JSON.stringify(value), { status, headers: { "Content-Type": "application/json" } });
  const bytesOf = async (content: unknown) => new Uint8Array(content instanceof Blob ? await content.arrayBuffer() : (content as ArrayBuffer));
  async function fetch(input: RequestInfo | URL, init: RequestInit = {}): Promise<Response> {
    const url = new URL(input instanceof Request ? input.url : String(input));
    const user = users[new Headers(init.headers).get("authorization")?.replace(/^Bearer /, "") ?? ""];
    if (url.host === "storage.example") {
      stored.set(url.pathname.slice(1), await bytesOf(init.body));
      return new Response(null, { status: 200 });
    }
    if (url.pathname === "/api/whoami-v2") {
      if (!user) return answer({ error: "Invalid credentials in Authorization header" }, 401);
      return answer({ type: "user", name: user, orgs: [{ type: "org", name: "cellmap" }], auth: { type: "access_token", accessToken: { displayName: "test", role: "write" } } });
    }
    if (!user) return answer({ error: "Invalid credentials" }, 401);
    if (refuse) return answer({ error: "You don't have the rights to do that" }, refuse);
    const sent = typeof init.body === "string" ? init.body : "";
    if (url.pathname === "/api/repos/create") {
      const { organization, name } = JSON.parse(sent);
      if (repos.has(`${organization}/${name}`)) return answer({ error: "You already created this model repo" }, 409);
      repos.add(`${organization}/${name}`);
      return answer({ url: `https://huggingface.co/${organization}/${name}`, id: "1" });
    }
    if (url.pathname.includes("/preupload/")) {
      const files = JSON.parse(sent).files as { path: string }[];
      return answer({ files: files.map((f) => ({ path: f.path, uploadMode: f.path.endsWith(".onnx") ? "lfs" : "regular" })) });
    }
    if (url.pathname.endsWith(".git/info/lfs/objects/batch")) {
      const objects = JSON.parse(sent).objects as { oid: string; size: number }[];
      return answer({ transfer: "basic", objects: objects.map((o) => ({ ...o, actions: { upload: { href: `https://storage.example/${o.oid}` } } })) });
    }
    if (url.pathname.includes("/commit/")) {
      const files = new Map<string, Uint8Array>();
      for (const line of sent.split("\n").map((l) => JSON.parse(l))) {
        if (line.key === "file") files.set(line.value.path, Uint8Array.from(atob(line.value.content), (c) => c.charCodeAt(0)));
        if (line.key === "lfsFile") files.set(line.value.path, stored.get(line.value.oid)!);
      }
      committed.push(files);
      return answer({ commitOid: "0123abc", commitUrl: `${url.origin}/commit/0123abc` });
    }
    return answer({ error: `no route ${url.pathname}` }, 404);
  }
  return { fetch, committed };
}

/** A context with the dataset open and one model finetuned from setup_54 in its catalog. */
async function withFinetuned(): Promise<{ ctx: FakeContext; base: ModelEntry }> {
  const ctx = fakeContext({ state: { datasetUrl: RAW } });
  const base = ctx.catalog.entries[0];
  const record = finetunedRecord(base, NAME, ONNX.byteLength, {
    volume: "",
    created_at: "2026-10-26T10:00:00.000Z",
    params: { finetune_mode: "head", num_epochs: 20 },
    epochs: 20,
    final_loss: 0.031,
  });
  await ctx.catalog.addFinetuned(record, ONNX.slice().buffer);
  return { ctx, base };
}

/** The publish's answer once it is no longer running. */
async function settled(router: ReturnType<typeof createRouter>, ctx: FakeContext): Promise<any> {
  for (let i = 0; i < 500; i++) {
    const answer = body(await router.handle(request("GET", `/api/finetune/model/${NAME}/publish`), ctx));
    if (answer.status !== "running") return answer;
    await new Promise((resolve) => setTimeout(resolve, 2));
  }
  throw new Error("the publish never finished");
}

afterEach(() => {
  vi.unstubAllGlobals();
});

describe("publishing a model finetuned in the browser", () => {
  it("commits the model card, the manifest and the ONNX file, and keeps where they went in the record", async () => {
    const hub = fakeHub({ hf_alice: "alice" });
    vi.stubGlobal("fetch", hub.fetch);
    const { ctx, base } = await withFinetuned();
    const router = createRouter({ publish: { now: () => new Date("2026-10-26T12:00:00.000Z") } });
    expect(body(await router.handle(request("POST", "/api/huggingface/credentials", { token: "hf_alice" }), ctx))).toMatchObject({ signed_in: true, user: "alice", via: "token" });

    const started = await router.handle(request("POST", `/api/finetune/model/${NAME}/publish`, { include_dataset: true }), ctx);
    expect(started.status).toBe(200);
    expect(body(started)).toMatchObject({ status: "running", repo: `alice/${NAME}`, folder: NAME });
    const done = await settled(router, ctx);
    const ref = `alice/${NAME}/${NAME}`;
    expect(done).toMatchObject({ status: "done", created: true, ref, url: `https://huggingface.co/alice/${NAME}`, try_link: `http://localhost/?model=${ref}` });
    expect(done.dataset_link).toBe(`http://localhost/?dataset=${RAW}&model=${ref}`);

    const [files] = hub.committed;
    expect([...files.keys()]).toEqual(["README.md", `${NAME}/manifest.json`, `${NAME}/${NAME}.onnx`]);
    expect([...files.get(`${NAME}/${NAME}.onnx`)!]).toEqual([...ONNX]);
    const manifest = JSON.parse(new TextDecoder().decode(files.get(`${NAME}/manifest.json`)));
    expect(manifest.published).toEqual({
      name: NAME,
      finetuned_from: base.id,
      finetune_mode: "head",
      epochs: 20,
      final_loss: 0.031,
      trained_at: "2026-10-26T10:00:00.000Z",
      published_at: "2026-10-26T12:00:00.000Z",
      app: expect.stringMatching(/^cellmap-flow \S+ \(browser\)$/),
      dataset: RAW,
    });
    // The catalog reads it as any model folder's manifest, with the trained file its one variant.
    const read = parseManifest(manifest, `${ref}/manifest.json`);
    expect(read.variants.map((v) => v.file)).toEqual([`${NAME}.onnx`]);
    expect(read.channels).toEqual(base.manifest.channels);
    // The card links to the public app, the model alone and on the dataset (asked for here).
    const card = new TextDecoder().decode(files.get("README.md"));
    expect(card.startsWith("---\nlibrary_name: onnx\n")).toBe(true);
    expect(card).toContain(`${PUBLIC_APP}?model=${ref}`);
    expect(card).toContain(`${PUBLIC_APP}?dataset=${RAW}&model=${ref}`);

    const [record] = await ctx.catalog.finetuned();
    expect(record.published).toEqual({ repo: `alice/${NAME}`, folder: NAME, at: "2026-10-26T12:00:00.000Z" });
    // A page that did not publish it (after a reload) answers from the record.
    expect(body(await createRouter().handle(request("GET", `/api/finetune/model/${NAME}/publish`), ctx))).toMatchObject({ status: "done", ref });

    // Again, without the dataset: a new commit to the same repository.
    await router.handle(request("POST", `/api/finetune/model/${NAME}/publish`, {}), ctx);
    expect(await settled(router, ctx)).toMatchObject({ status: "done", created: false, ref });
    expect(hub.committed).toHaveLength(2);
    const again = JSON.parse(new TextDecoder().decode(hub.committed[1].get(`${NAME}/manifest.json`)));
    expect(again.published).not.toHaveProperty("dataset");
    expect(new TextDecoder().decode(hub.committed[1].get("README.md"))).not.toContain(RAW);
  });

  it("refuses without a token, a token Hugging Face does not take, and an owner that is not the user", async () => {
    const hub = fakeHub({ hf_alice: "alice" });
    vi.stubGlobal("fetch", hub.fetch);
    const { ctx } = await withFinetuned();
    const router = createRouter();
    expect((await router.handle(request("POST", `/api/finetune/model/${NAME}/publish`, {}), ctx)).status).toBe(401);
    expect((await router.handle(request("POST", "/api/huggingface/credentials", { token: "hf_revoked" }), ctx)).status).toBe(401);
    expect(body(await router.handle(request("GET", "/api/huggingface/credentials"), ctx))).toMatchObject({ signed_in: false, oauth_url: null });

    await router.handle(request("POST", "/api/huggingface/credentials", { token: "hf_alice" }), ctx);
    expect((await router.handle(request("POST", `/api/finetune/model/${NAME}/publish`, { repo: `bob/${NAME}` }), ctx)).status).toBe(403);
    // One of the user's organizations is theirs to publish under.
    await router.handle(request("POST", `/api/finetune/model/${NAME}/publish`, { repo: `cellmap/${NAME}` }), ctx);
    expect(await settled(router, ctx)).toMatchObject({ status: "done", ref: `cellmap/${NAME}/${NAME}` });

    await router.handle(request("POST", "/api/huggingface/credentials", { forget: true }), ctx);
    expect((await router.handle(request("POST", `/api/finetune/model/${NAME}/publish`, {}), ctx)).status).toBe(401);
  });

  it("fails the publish, keeping no record of it, when the token may not write", async () => {
    const hub = fakeHub({ hf_alice: "alice" }, 403);
    vi.stubGlobal("fetch", hub.fetch);
    const { ctx } = await withFinetuned();
    const router = createRouter();
    await router.handle(request("POST", "/api/huggingface/credentials", { token: "hf_alice" }), ctx);
    await router.handle(request("POST", `/api/finetune/model/${NAME}/publish`, {}), ctx);
    const failed = await settled(router, ctx);
    expect(failed.status).toBe("failed");
    expect(failed.error).toEqual(expect.any(String));
    expect(failed.error).not.toContain("hf_alice");
    expect(hub.committed).toHaveLength(0);
    expect((await ctx.catalog.finetuned())[0].published).toBeUndefined();
  });
});

describe("Sign in with Hugging Face", () => {
  it("takes the token the popup hands over on the page's channel, and tells the page", async () => {
    vi.stubGlobal("fetch", fakeHub({ hf_oauth_alice: "alice" }).fetch);
    const { ctx } = await withFinetuned();
    const page = new EventTarget();
    const router = createRouter({ publish: { clientId: "test-client", page: () => page } });
    const url = new URL(body(await router.handle(request("GET", "/api/huggingface/credentials"), ctx)).oauth_url);
    expect(url.pathname).toBe("/oauth.html");

    const told = new Promise((resolve) => page.addEventListener(SIGN_IN_EVENT, (e) => resolve((e as CustomEvent).detail), { once: true }));
    handOver(url.searchParams.get("channel")!, { token: "hf_oauth_alice", expires_at: "2099-01-01T00:00:00.000Z" });
    expect(await told).toEqual({ signed_in: true });
    expect(body(await router.handle(request("GET", "/api/huggingface/credentials"), ctx))).toMatchObject({
      signed_in: true,
      user: "alice",
      via: "oauth",
      expires_at: "2099-01-01T00:00:00.000Z",
    });
  });
});
