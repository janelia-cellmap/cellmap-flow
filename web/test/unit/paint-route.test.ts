// The paint route (src/state/paint-route.ts) on a memory store: the S3
// semantics the Neuroglancer fork's s3+http kvstore and its zarr source need
// of the service worker, without a worker.
import { describe, expect, it } from "vitest";

import { PAINT_METHODS, servePaint, type PaintRequest, type PaintResponse } from "../../src/state/paint-route";
import { MemoryPaintStore } from "../../src/state/paint-store";

const VOL = "vol-0123abcd-20261015-120000";
const CHUNK = `${VOL}/annotation/s0/3.4.5`;

function req(method: string, path: string, extra: Partial<PaintRequest> = {}): PaintRequest {
  return { method, path, body: () => Promise.reject(new Error("no body was sent")), ...extra };
}

function withBody(bytes: Uint8Array): Pick<PaintRequest, "body"> {
  return { body: async () => bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength) as ArrayBuffer };
}

const body = (response: PaintResponse) => (response.body === null ? null : Array.from(response.body));
const textOf = (response: PaintResponse) => new TextDecoder().decode(response.body ?? new Uint8Array());

/** A store with one 10-byte chunk and the volume's metadata. */
async function painted() {
  const store = new MemoryPaintStore();
  await store.put(VOL, "annotation/s0/3.4.5", new Uint8Array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]));
  await store.put(VOL, "annotation/s0/.zarray", new TextEncoder().encode('{"zarr_format":2}'));
  return store;
}

describe("servePaint: GET and HEAD", () => {
  it("gives an object whole, with its length, as a stream of bytes", async () => {
    const response = await servePaint(await painted(), req("GET", CHUNK));
    expect(response.status).toBe(200);
    expect(body(response)).toEqual([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]);
    expect(response.headers["content-type"]).toBe("application/octet-stream");
    expect(response.headers["content-length"]).toBe("10");
    expect(response.headers["accept-ranges"]).toBe("bytes");
    expect(response.headers.etag).toMatch(/^".+"$/);
  });

  it("gives zarr v2 metadata as JSON", async () => {
    const store = await painted();
    expect((await servePaint(store, req("GET", `${VOL}/annotation/s0/.zarray`))).headers["content-type"]).toBe("application/json");
    await store.put(VOL, ".zattrs", new Uint8Array([123, 125]));
    await store.put(VOL, "annotation/.zgroup", new Uint8Array([123, 125]));
    expect((await servePaint(store, req("GET", `${VOL}/.zattrs`))).headers["content-type"]).toBe("application/json");
    expect((await servePaint(store, req("HEAD", `${VOL}/annotation/.zgroup`))).headers["content-type"]).toBe("application/json");
  });

  it("is a 404 for a chunk never painted (the fill value to a zarr reader), with no body on HEAD", async () => {
    const store = await painted();
    const missing = await servePaint(store, req("GET", `${VOL}/annotation/s0/0.0.0`));
    expect(missing.status).toBe(404);
    const head = await servePaint(store, req("HEAD", `${VOL}/annotation/s0/0.0.0`));
    expect(head.status).toBe(404);
    expect(head.body).toBeNull();
  });

  it("answers HEAD with GET's headers and no body (the fork's stat)", async () => {
    const response = await servePaint(await painted(), req("HEAD", CHUNK));
    expect(response.status).toBe(200);
    expect(response.body).toBeNull();
    expect(response.headers["content-length"]).toBe("10");
    expect(response.headers["content-type"]).toBe("application/octet-stream");
  });

  it("honours one byte range (what the fork's reader sends) as a 206 with content-range", async () => {
    const store = await painted();
    const part = await servePaint(store, req("GET", CHUNK, { range: "bytes=2-4" }));
    expect(part.status).toBe(206);
    expect(body(part)).toEqual([2, 3, 4]);
    expect(part.headers["content-range"]).toBe("bytes 2-4/10");
    expect(part.headers["content-length"]).toBe("3");
    const clipped = await servePaint(store, req("GET", CHUNK, { range: "bytes=8-100" }));
    expect(body(clipped)).toEqual([8, 9]);
    expect(clipped.headers["content-range"]).toBe("bytes 8-9/10");
    const open = await servePaint(store, req("GET", CHUNK, { range: "bytes=7-" }));
    expect(body(open)).toEqual([7, 8, 9]);
    const suffix = await servePaint(store, req("GET", CHUNK, { range: "bytes=-3" }));
    expect(body(suffix)).toEqual([7, 8, 9]);
    expect(suffix.headers["content-range"]).toBe("bytes 7-9/10");
    const head = await servePaint(store, req("HEAD", CHUNK, { range: "bytes=0-0" }));
    expect(head.status).toBe(206);
    expect(head.body).toBeNull();
  });

  it("is a 416 for a range past the end: the fork probes an empty object with bytes=0-0 and expects one", async () => {
    const store = await painted();
    await store.put(VOL, "empty", new Uint8Array());
    const empty = await servePaint(store, req("GET", `${VOL}/empty`, { range: "bytes=0-0" }));
    expect(empty.status).toBe(416);
    expect(empty.headers["content-range"]).toBe("bytes */0");
    expect(empty.body).toBeNull();
    const past = await servePaint(store, req("GET", CHUNK, { range: "bytes=10-10" }));
    expect(past.status).toBe(416);
    expect(past.headers["content-range"]).toBe("bytes */10");
  });

  it("ignores a Range it does not understand and answers the whole object, as HTTP allows", async () => {
    const store = await painted();
    for (const range of ["bytes=0-1,4-5", "items=0-1", "bytes=5-2", "bytes=-"]) {
      const response = await servePaint(store, req("GET", CHUNK, { range }));
      expect(response.status, range).toBe(200);
      expect(body(response), range).toHaveLength(10);
    }
  });
});

describe("servePaint: PUT and DELETE (the fork's write and delete)", () => {
  it("stores a PUT's body and gives it back on GET, byte for byte; the PUT is a 200 with an empty body", async () => {
    const store = new MemoryPaintStore();
    const painted = new Uint8Array([2, 2, 0, 1, 1, 0, 0, 2]);
    const put = await servePaint(store, req("PUT", CHUNK, withBody(painted)));
    expect(put.status).toBe(200);
    expect(put.body).toBeNull();
    expect(put.headers.etag).toMatch(/^"[0-9a-f]+-[0-9a-f]{8}"$/);
    const got = await servePaint(store, req("GET", CHUNK));
    expect(body(got)).toEqual([2, 2, 0, 1, 1, 0, 0, 2]);
    expect(got.headers.etag).toBe(put.headers.etag);  // the same object, the same tag
    expect(Array.from((await store.get(VOL, "annotation/s0/3.4.5"))!)).toEqual([2, 2, 0, 1, 1, 0, 0, 2]);
    // The brush paints again: the new bytes replace the old.
    await servePaint(store, req("PUT", CHUNK, withBody(new Uint8Array([1]))));
    expect(body(await servePaint(store, req("GET", CHUNK)))).toEqual([1]);
  });

  it("is a 204 on DELETE whether or not the object was there, and the object is gone", async () => {
    const store = await painted();
    expect((await servePaint(store, req("DELETE", CHUNK))).status).toBe(204);
    expect(await store.get(VOL, "annotation/s0/3.4.5")).toBeUndefined();
    expect((await servePaint(store, req("GET", CHUNK))).status).toBe(404);
    const again = await servePaint(store, req("DELETE", CHUNK));
    expect(again.status).toBe(204);
    expect(again.body).toBeNull();
  });

  it("reads and writes a volume spelled with the viewer's ~<n> refresh suffix as the volume itself", async () => {
    // viewer/layers.ts respelledPaintUrl gives the paint layer's source a new
    // spelling so Neuroglancer re-reads it; the store knows one volume.
    const store = await painted();
    for (const spelled of [`${VOL}~1`, `${VOL}~2`, `${VOL}~10`, `${VOL}~0`]) {
      const got = await servePaint(store, req("GET", `${spelled}/annotation/s0/3.4.5`));
      expect(got.status, spelled).toBe(200);
      expect(body(got), spelled).toEqual([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]);
      expect((await servePaint(store, req("HEAD", `${spelled}/annotation/s0/.zarray`))).status, spelled).toBe(200);
    }
    await servePaint(store, req("PUT", `${VOL}~3/annotation/s0/0.0.0`, withBody(new Uint8Array([7]))));
    expect(Array.from((await store.get(VOL, "annotation/s0/0.0.0"))!)).toEqual([7]);  // under the volume, not a "~3" one
    expect(await store.volumes()).toEqual([VOL]);
    expect((await servePaint(store, req("DELETE", `${VOL}~1/annotation/s0/0.0.0`))).status).toBe(204);
    expect(await store.get(VOL, "annotation/s0/0.0.0")).toBeUndefined();
  });

  it("decodes a percent-encoded path, so the store's key is the file's own name", async () => {
    const store = new MemoryPaintStore();
    await servePaint(store, req("PUT", `${VOL}/a%20b/.zattrs`, withBody(new Uint8Array([1]))));
    expect(await store.list(VOL)).toEqual(["a b/.zattrs"]);
    expect((await servePaint(store, req("GET", `${VOL}/a b/.zattrs`))).status).toBe(200);
  });
});

describe("servePaint: what is refused", () => {
  it("is a 405 with an allow header for any other method", async () => {
    const store = await painted();
    for (const method of ["POST", "PATCH", "OPTIONS"]) {
      const response = await servePaint(store, req(method, CHUNK));
      expect(response.status, method).toBe(405);
      expect(response.headers.allow).toBe(PAINT_METHODS);
      expect(PAINT_METHODS).toBe("GET, HEAD, PUT, DELETE");
    }
  });

  it("is a 400 for a path with no volume, no file, or a '.' or '..' segment", async () => {
    const store = await painted();
    for (const path of ["", "/", `/${VOL}/.zattrs`, VOL, `${VOL}/`, `${VOL}/annotation/../.zattrs`, `${VOL}/./.zattrs`, `${VOL}/annotation//x`, `../${VOL}/x`, `${VOL}/%E0%A4%A`]) {
      const response = await servePaint(store, req("GET", path));
      expect(response.status, JSON.stringify(path)).toBe(400);
      expect(textOf(response), JSON.stringify(path)).not.toBe("");
    }
    // A PUT is checked the same way: nothing lands under such a key.
    const put = await servePaint(store, req("PUT", `${VOL}/../x`, withBody(new Uint8Array([1]))));
    expect(put.status).toBe(400);
    expect(await store.volumes()).toEqual([VOL]);
  });

  it("is a 400 for a '~' in the volume that is not a ~<digits> suffix, which no volume id has", async () => {
    const store = await painted();
    for (const spelled of [`${VOL}~`, `${VOL}~x`, `${VOL}~1a`, `~1`, `vol~1-abc`, `${VOL}~1~2`, `${VOL}~-1`]) {
      const response = await servePaint(store, req("GET", `${spelled}/annotation/s0/3.4.5`));
      expect(response.status, spelled).toBe(400);
      expect(textOf(response), spelled).toContain("~");
    }
    // Nothing lands under such a spelling either.
    expect((await servePaint(store, req("PUT", `${VOL}~/x`, withBody(new Uint8Array([1]))))).status).toBe(400);
    expect(await store.volumes()).toEqual([VOL]);
    // A "~" in the path within the volume is just a character of a key, as before.
    expect((await servePaint(store, req("GET", `${VOL}/annotation/s0/3~4`))).status).toBe(404);
  });

  it("is a 404 for any request with a query (an S3 listing probe), even of an object that is there", async () => {
    const store = await painted();
    expect((await servePaint(store, req("GET", `${VOL}/annotation/s0/`, { query: "?list-type=2&prefix=annotation%2Fs0%2F&delimiter=%2F" }))).status).toBe(404);
    expect((await servePaint(store, req("GET", CHUNK, { query: "?x=1" }))).status).toBe(404);
    expect((await servePaint(store, req("GET", CHUNK, { query: "" }))).status).toBe(200);  // URL.search of a plain URL
  });
});
