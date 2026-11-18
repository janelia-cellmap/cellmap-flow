// JSON requests to the dashboard's own routes.
//
// getJSON and postJSON resolve to the parsed body of a 2xx response. They
// throw an ApiError for any other status, with the body's "error" as the
// message (or "HTTP <status>" when there is none, e.g. an HTML error page),
// and for a 2xx answer that is not JSON. A request that never got an answer
// rejects with fetch's own error (TypeError, or AbortError after an abort),
// unchanged, so callers can tell "the server said no" from "no server".

export class ApiError extends Error {
  constructor(message, status, body) {
    super(message);
    this.name = "ApiError";
    this.status = status;
    this.body = body;  // the parsed JSON body, or null
  }
}

async function requestJSON(url, init) {
  const response = await fetch(url, init);
  let body;
  try {
    body = await response.json();
  } catch {
    body = undefined;
  }
  if (!response.ok) {
    const error = body && typeof body === "object" ? body.error : undefined;
    throw new ApiError(error ? String(error) : `HTTP ${response.status}`,
                       response.status, body === undefined ? null : body);
  }
  if (body === undefined) {
    throw new ApiError(`HTTP ${response.status}: the answer is not JSON`, response.status, null);
  }
  return body;
}

export function getJSON(url, { signal } = {}) {
  return requestJSON(url, { signal });
}

// body is sent as JSON; with no body (undefined) the request has none, and no
// Content-Type either.
export function postJSON(url, body, { signal, method = "POST" } = {}) {
  const init = { method, signal };
  if (body !== undefined) {
    init.headers = { "Content-Type": "application/json" };
    init.body = JSON.stringify(body);
  }
  return requestJSON(url, init);
}
