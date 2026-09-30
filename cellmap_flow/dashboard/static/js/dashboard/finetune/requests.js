// Requests to the finetune routes (dashboard/routes/finetune/).
//
// Those routes answer JSON with {success, error} at any HTTP status -- a 404
// for a job that is gone, a 400 for bad parameters -- and this tab shows a
// failed answer the same way whatever its status. So these resolve to the
// parsed answer for every status, where lib/api's getJSON and postJSON throw
// on a non-2xx one. A request that got no answer, or an answer that is not
// JSON, rejects as fetch and Response.json() do.

export function getAnswer(url) {
  return fetch(url).then((response) => response.json());
}

// The body is sent as JSON. With no body (undefined) the request has none,
// and no Content-Type either.
export function postAnswer(url, body) {
  const init = { method: "POST" };
  if (body !== undefined) {
    init.headers = { "Content-Type": "application/json" };
    init.body = JSON.stringify(body);
  }
  return fetch(url, init).then((response) => response.json());
}
