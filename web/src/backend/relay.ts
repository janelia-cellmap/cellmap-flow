// The page side of the service worker relay for /api/* requests. STUB.
import type { Backend } from "../contracts";

/** Register sw.js (scope ./) and resolve once it controls this page. */
export async function registerServiceWorker(): Promise<void> {
  throw new Error("backend/relay.ts is not implemented yet");
}

/** Answer the /api/* requests the service worker relays, through `backend`. */
export function listenForApiRequests(_backend: Backend): void {}
