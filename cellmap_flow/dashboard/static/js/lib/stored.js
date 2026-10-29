// The dashboard's values kept in this browser's localStorage, or with
// `storage` "session" in the tab's sessionStorage, which lasts through the
// tab's reloads and ends with it. Storage that cannot be used (a private
// window, blocked site data) reads as the fallback.
function storageOf(storage) {
  return storage === "session" ? window.sessionStorage : window.localStorage;
}

export function readText(key, fallback, storage = "local") {
  try {
    return storageOf(storage).getItem(key) ?? fallback;
  } catch {
    return fallback;
  }
}

export function writeText(key, text, storage = "local") {
  try {
    storageOf(storage).setItem(key, text);
  } catch {}  // not kept
}

// A missing, null or unparsable value reads as the fallback.
export function readJSON(key, fallback, storage = "local") {
  try {
    return JSON.parse(readText(key, "null", storage)) ?? fallback;
  } catch {
    return fallback;
  }
}

export function writeJSON(key, value, storage = "local") {
  writeText(key, JSON.stringify(value), storage);
}
