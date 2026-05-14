// Registers the /sw.js virtual-zarr service worker and routes `vz-request`
// messages to the per-tab handler. The handler is set by activateVz() in
// vz-handler.ts; until then, /vz/* returns 503.

import { handleVzRequest } from "./vz-handler";

export async function registerVirtualZarrSW(): Promise<ServiceWorkerRegistration> {
  if (!("serviceWorker" in navigator)) {
    throw new Error(
      "navigator.serviceWorker is undefined. " +
        "Service workers require a secure context (https, localhost, or whitelisted origin). " +
        `Observed: location.origin=${location.origin}, isSecureContext=${window.isSecureContext}.`,
    );
  }

  const reg = await navigator.serviceWorker.register("/sw.js");
  await navigator.serviceWorker.ready;

  if (!navigator.serviceWorker.controller) {
    // First install at this origin: skipWaiting() + clients.claim() in
    // sw.js should make the new SW take control without a reload, but the
    // controllerchange event can lag a tick. Wait up to 5s for it; if it
    // never fires (very rare), fall through anyway — /vz/ will return 503
    // and the next Stream click will pick up a controlled page.
    await new Promise<void>((resolve) => {
      const onCtrl = () => {
        navigator.serviceWorker.removeEventListener("controllerchange", onCtrl);
        resolve();
      };
      navigator.serviceWorker.addEventListener("controllerchange", onCtrl);
      setTimeout(() => {
        navigator.serviceWorker.removeEventListener("controllerchange", onCtrl);
        resolve();
      }, 5000);
    });
  }

  navigator.serviceWorker.addEventListener("message", async (event) => {
    const data = event.data;
    if (!data || data.type !== "vz-request") return;
    const port = event.ports[0];
    if (!port) return;
    try {
      const res = await handleVzRequest(String(data.path ?? ""));
      port.postMessage(res, res.body instanceof ArrayBuffer ? [res.body] : []);
    } catch (err) {
      const e = err as Error;
      port.postMessage({
        status: 500,
        headers: { "content-type": "text/plain" },
        body: `${e.message}\n${e.stack ?? ""}`,
      });
    }
  });

  return reg;
}
