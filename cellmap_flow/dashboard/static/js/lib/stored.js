// What the dashboard keeps in this browser between visits, in its
// localStorage: a convenience of this browser alone, gone with its site
// data. Storage that cannot be read or written (a private window, blocked
// site data, a full quota) is not an error: a read gives the fallback and a
// write keeps nothing, so the page works as it does with nothing kept.
//
// readJSON and writeJSON keep a value as JSON; readText and writeText keep a
// string as it is, for the keys that have always held a plain string.

// The string kept under `key`; `fallback` when there is none, or the storage
// cannot be read.
export function readText(key, fallback) {
  try {
    const text = window.localStorage.getItem(key);
    return text === null ? fallback : text;
  } catch {
    return fallback;
  }
}

// Keep the string `text` under `key`; nothing when the storage cannot be
// written, and the next visit finds what was kept before.
export function writeText(key, text) {
  try {
    window.localStorage.setItem(key, text);
  } catch {
    // Not kept.
  }
}

// The value kept under `key`, parsed from JSON; `fallback` when there is
// none, or it cannot be read or parsed.
export function readJSON(key, fallback) {
  const text = readText(key, null);
  if (text === null) return fallback;
  try {
    return JSON.parse(text);
  } catch {
    return fallback;
  }
}

// Keep `value` under `key`, as JSON; nothing when the storage cannot be
// written.
export function writeJSON(key, value) {
  writeText(key, JSON.stringify(value));
}
