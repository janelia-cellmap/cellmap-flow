// The /api/* backend: routes the dashboard's requests to handlers with the same
// JSON contract as the Flask routes. STUB: replaced by the backend module.
import type { AppContext, Backend } from "../contracts";

export function createBackend(_ctx: AppContext): Backend {
  return {
    async handle(req) {
      return { status: 501, json: { success: false, available: false, error: `not implemented in the browser: ${req.method} ${req.path}` } };
    },
  };
}
