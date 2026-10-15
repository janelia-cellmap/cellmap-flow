// The service worker: relays /api/* to the page's backend and /virtual/* to
// chunkmirage's Engine, serves /local/ files. STUB: replaced by the backend module.
/// <reference lib="webworker" />
declare const self: ServiceWorkerGlobalScope;

self.addEventListener("install", () => self.skipWaiting());
self.addEventListener("activate", (event) => event.waitUntil(self.clients.claim()));
