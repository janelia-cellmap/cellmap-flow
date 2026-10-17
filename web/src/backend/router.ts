// Routing for the /api/* backend: a method and a path pattern per handler,
// as the Flask blueprints have (`/api/finetune/job/<job_id>/status` is
// `/api/finetune/job/:id/status` here), and the few ways a route answers.
//
// Flask's conventions, kept so the dashboard's JavaScript sees no difference:
// a JSON body is answered with a JSON body; a bad request is a 400
// `{success: false, error}` (dashboard/requests.py's parse); a method the
// path does not take is a 405. What has no browser implementation answers
// `unavailable()`: a 503 with `{success: false, available: false, error}`
// (the Unavailable contract), which lib/api.js turns into an ApiError whose
// message is the reason, so the page can show it where it would show any
// other failure.
import type { ApiRequest, ApiResponse, AppContext, RouteHandler, Unavailable } from "../contracts";

export type Method = "GET" | "POST" | "PUT" | "DELETE" | "PATCH";

interface Route {
  method: Method;
  /** The pattern's segments: a literal, or a parameter name after ":". */
  segments: string[];
  handler: RouteHandler;
}

export interface Match {
  handler: RouteHandler;
  params: Record<string, string>;
}

export class Router {
  private routes: Route[] = [];
  private fallback: RouteHandler = (req) => unavailable(`${req.method} ${req.path} is not a route of the dashboard`);

  /** Register `handler` for `method` on `pattern` ("/api/finetune/job/:id/status"). */
  add(method: Method, pattern: string, handler: RouteHandler): this {
    this.routes.push({ method, segments: split(pattern), handler });
    return this;
  }

  get(pattern: string, handler: RouteHandler): this {
    return this.add("GET", pattern, handler);
  }

  post(pattern: string, handler: RouteHandler): this {
    return this.add("POST", pattern, handler);
  }

  put(pattern: string, handler: RouteHandler): this {
    return this.add("PUT", pattern, handler);
  }

  delete(pattern: string, handler: RouteHandler): this {
    return this.add("DELETE", pattern, handler);
  }

  /** The handler for a request no route matches (by default: unavailable). */
  otherwise(handler: RouteHandler): this {
    this.fallback = handler;
    return this;
  }

  /** The route for `method` on `path`, with the pattern's parameters filled
   * in; null when no pattern matches the path; `{methods}` when one does but
   * not for this method. A trailing slash does not count. */
  match(method: string, path: string): Match | { methods: Method[] } | null {
    const parts = split(path);
    const allowed: Method[] = [];
    for (const route of this.routes) {
      const params = bind(route.segments, parts);
      if (!params) continue;
      if (route.method === method) return { handler: route.handler, params };
      allowed.push(route.method);
    }
    return allowed.length ? { methods: allowed } : null;
  }

  async handle(req: ApiRequest, ctx: AppContext): Promise<ApiResponse> {
    const found = this.match(req.method, req.path);
    if (found && "methods" in found) {
      return json(405, { success: false, error: `${req.method} is not allowed for ${req.path} (${found.methods.join(", ")})` });
    }
    if (!found) return this.fallback(req, {}, ctx);
    return found.handler(req, found.params, ctx);
  }
}

/** A path's segments, without the empty ones a leading or trailing slash gives. */
function split(path: string): string[] {
  return path.split("/").filter((s) => s !== "");
}

/** The parameters `segments` bind to `parts`, or null when they do not match. */
function bind(segments: string[], parts: string[]): Record<string, string> | null {
  if (segments.length !== parts.length) return null;
  const params: Record<string, string> = {};
  for (let i = 0; i < segments.length; i++) {
    const s = segments[i];
    if (s.startsWith(":")) {
      try {
        params[s.slice(1)] = decodeURIComponent(parts[i]);
      } catch {
        params[s.slice(1)] = parts[i];
      }
    } else if (s !== parts[i]) return null;
  }
  return params;
}

// ------------------------------------------------ answers

export function json(status: number, value: unknown): ApiResponse {
  return { status, json: value };
}

export function text(status: number, body: string | ArrayBuffer, type = "text/plain; charset=utf-8", headers?: Record<string, string>): ApiResponse {
  return headers ? { status, body, type, headers } : { status, body, type };
}

/** The 400 every Flask route answers a bad body with (dashboard/requests.py). */
export function badRequest(error: string): ApiResponse {
  return json(400, { success: false, error });
}

/** What a route only a server can serve answers: `reason` says why, in words
 * the page can show. */
export function unavailable(reason: string): ApiResponse {
  const body: Unavailable = { success: false, available: false, error: reason };
  return json(503, body);
}

/** The request's body when it is a JSON object, else null (a route then
 * answers badRequest("expected a JSON object"), as parse() does). */
export function objectBody(req: ApiRequest): Record<string, unknown> | null {
  const body = req.body;
  return body && typeof body === "object" && !Array.isArray(body) ? (body as Record<string, unknown>) : null;
}

/** An error's message, for an answer's `error`. */
export { message } from "../shared/message";
