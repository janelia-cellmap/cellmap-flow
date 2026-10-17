// The dashboard's values kept in this browser's localStorage. Storage that
// cannot be used (a private window, blocked site data) reads as the fallback.
export function readText(key, fallback) {
  try {
    return window.localStorage.getItem(key) ?? fallback;
  } catch {
    return fallback;
  }
}

export function writeText(key, text) {
  try {
    window.localStorage.setItem(key, text);
  } catch {}  // not kept
}

// A missing, null or unparsable value reads as the fallback.
export function readJSON(key, fallback) {
  try {
    return JSON.parse(readText(key, "null")) ?? fallback;
  } catch {
    return fallback;
  }
}

export function writeJSON(key, value) {
  writeText(key, JSON.stringify(value));
}
