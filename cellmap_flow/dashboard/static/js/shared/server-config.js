// Checking and saving the LSF server config, for the first-run dialog on
// the dashboard page and the Models tab's Server Config section.
//
// /api/server-config stores what it is sent. The count fields come from
// <input type="number"> as strings, and a blank one ("" before a form has
// loaded, or cleared by hand) reached int("") on the server and a 500. So a
// count goes as a whole number, a blank one is left out so the server keeps
// its value, and anything else is refused here with a message.
import { ApiError, postJSON } from "../lib/api.js";

// The count typed in `raw`, or undefined when it is blank. Throws an Error
// saying what is wrong, naming the field by `label`.
export function readCount(raw, label) {
  const text = String(raw).trim();
  if (text === "") return undefined;
  const n = Number(text);
  if (!Number.isInteger(n) || n < 1) {
    throw new Error(label + " must be a whole number of at least 1.");
  }
  return n;
}

// LSF's run limit, HH:MM or minutes: the rule jobs.lsf applies before it
// passes -W to bsub (it drops a value it cannot parse, and the job gets the
// queue's default).
export const WALLTIME_RE = /^\d+(:\d{1,2})?$/;

// Save `payload`. Resolves to the server's answer once it has stored it;
// otherwise throws an ApiError whose message is the server's reason, or
// "HTTP <status>". A request that got no answer rejects with fetch's error.
export async function saveServerConfig(payload) {
  const data = await postJSON("/api/server-config", payload);
  if (!data || !data.success) {
    // A 2xx answer without success: the route itself never gives one.
    throw new ApiError((data && data.error) || "HTTP 200", 200, data);
  }
  return data;
}
