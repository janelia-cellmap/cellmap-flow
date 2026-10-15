// sw.js — CellMap Flow serverless service worker.
//
// Intercepts Neuroglancer's same-origin Zarr requests. It does NOT run the
// inference itself: service workers forbid dynamic import() (zarrita/ORT use it),
// so each /cf/ request is delegated to the page's compute Worker via postMessage
// and the result is relayed back as the fetch response.
//
// Routing (all same-origin, under the SW scope):
//   .../cf/<model[__CFLOW_ARGS__<b64>__CFLOW_ARGS__]>/{.zattrs, s0/.zarray, s0/z.y.x[.c]}
//                                                       -> delegate to compute Worker
//   .../local/<handleKey>/<path...>                     -> raw local Zarr file (read here)
//
// Register from the page with { type: "module" } so the import below works.

import { getHandle, FileSystemStore } from "./local-store.js";

self.addEventListener("install", () => self.skipWaiting());
self.addEventListener("activate", (e) => e.waitUntil(self.clients.claim()));

const CORS = { "Access-Control-Allow-Origin": "*" };

// Pick the app window to delegate to (not the Neuroglancer iframe).
async function pickAppClient() {
  const all = await self.clients.matchAll({ type: "window", includeUncontrolled: true });
  return all.find((c) => !c.url.includes("/vendor/neuroglancer/")) || all[0];
}

// Ask the page's compute Worker to produce a response for a /cf/ request.
function delegateCf(segment, tail) {
  return new Promise(async (resolve, reject) => {
    const client = await pickAppClient();
    if (!client) {
      reject(new Error("no app window open to compute chunks"));
      return;
    }
    const { port1, port2 } = new MessageChannel();
    port1.onmessage = (e) =>
      e.data.ok ? resolve(e.data) : reject(new Error(e.data.error || "compute failed"));
    client.postMessage({ type: "computeCf", segment, tail }, [port2]);
  });
}

// Raw passthrough for a locally-picked Zarr directory: .../local/<handleKey>/<path>
async function handleLocalFile(handleKey, path) {
  const dirHandle = await getHandle(handleKey);
  if (!dirHandle) return new Response("no handle", { status: 404 });
  const bytes = await new FileSystemStore(dirHandle).get(path);
  if (bytes === undefined) return new Response("not found", { status: 404 });
  const isMeta = /\.z(array|attrs|group)$|zarr\.json$/.test(path);
  return new Response(bytes, {
    headers: { "Content-Type": isMeta ? "application/json" : "application/octet-stream", ...CORS },
  });
}

async function route(url) {
  const lidx = url.pathname.indexOf("/local/");
  if (lidx !== -1) {
    const rest = url.pathname.slice(lidx + "/local/".length).split("/");
    return handleLocalFile(rest[0], rest.slice(1).join("/"));
  }

  const cidx = url.pathname.indexOf("/cf/");
  if (cidx === -1) return null; // not ours
  const rest = url.pathname.slice(cidx + "/cf/".length);
  const slash = rest.indexOf("/");
  const segment = slash === -1 ? rest : rest.slice(0, slash);
  const tail = slash === -1 ? "" : rest.slice(slash + 1);

  const r = await delegateCf(segment, tail);
  return new Response(r.body, {
    status: r.status || 200,
    headers: { "Content-Type": r.contentType, ...CORS },
  });
}

self.addEventListener("fetch", (event) => {
  const url = new URL(event.request.url);
  if (url.origin !== self.location.origin) return; // only same-origin
  if (!url.pathname.includes("/cf/") && !url.pathname.includes("/local/")) return;
  event.respondWith(
    route(url)
      .then((r) => r || fetch(event.request))
      .catch((e) => {
        console.error("[sw]", e);
        return new Response(String(e), { status: 500 });
      })
  );
});
