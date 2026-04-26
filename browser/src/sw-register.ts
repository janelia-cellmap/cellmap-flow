import { handleRequest } from "./virtual-zarr";

export async function registerVirtualZarrSW(): Promise<ServiceWorkerRegistration> {
  if (!("serviceWorker" in navigator)) {
    throw new Error(
      "navigator.serviceWorker is undefined. " +
        `Service workers require a secure context (https, localhost, or whitelisted origin). ` +
        `Observed: location.origin=${location.origin}, isSecureContext=${window.isSecureContext}.`,
    );
  }

  const reg = await navigator.serviceWorker.register("/sw.js");
  await navigator.serviceWorker.ready;

  if (!navigator.serviceWorker.controller) {
    // First install at this origin: clients.claim() doesn't always retroactively
    // control the current page. Reload once so the next load is controlled.
    location.reload();
    await new Promise(() => {}); // unreachable
  }

  navigator.serviceWorker.addEventListener("message", async (event) => {
    const data = event.data;
    if (!data || data.type !== "vz-request") return;
    const port = event.ports[0];
    if (!port) return;
    try {
      const res = await handleRequest(String(data.path ?? ""));
      port.postMessage(res);
    } catch (err) {
      const e = err as Error;
      console.error(`[vz handler] ${data.path}:`, e);
      port.postMessage({
        status: 500,
        headers: { "content-type": "text/plain" },
        body: `${e.message}\n${e.stack ?? ""}`,
      });
    }
  });

  return reg;
}
