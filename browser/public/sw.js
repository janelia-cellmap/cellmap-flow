// cellmap-flow virtual-zarr service worker.
// Intercepts same-origin fetches under /vz/ and forwards them to an active
// page client via MessageChannel. The page holds the ORT session + raw
// zarr state and replies with bytes.

const PREFIX = "/vz/";

self.addEventListener("install", () => self.skipWaiting());
self.addEventListener("activate", (e) => e.waitUntil(self.clients.claim()));

self.addEventListener("fetch", (event) => {
  const url = new URL(event.request.url);
  if (url.origin !== self.location.origin) return;
  if (!url.pathname.startsWith(PREFIX)) return;
  event.respondWith(handle(url));
});

async function handle(url) {
  const path = url.pathname.slice(PREFIX.length);

  const clients = await self.clients.matchAll({
    type: "window",
    includeUncontrolled: true,
  });
  const client = clients[0];
  if (!client) {
    return new Response("no active page client for /vz/", {
      status: 503,
      headers: { "content-type": "text/plain" },
    });
  }

  const channel = new MessageChannel();
  const reply = new Promise((resolve) => {
    channel.port1.onmessage = (ev) => resolve(ev.data);
    setTimeout(
      () => resolve({ status: 504, body: "page timeout", headers: {} }),
      120_000,
    );
  });

  client.postMessage({ type: "vz-request", path }, [channel.port2]);

  const res = await reply;
  const headers = {
    "access-control-allow-origin": "*",
    ...(res.headers || {}),
  };
  return new Response(res.body ?? null, { status: res.status ?? 500, headers });
}
