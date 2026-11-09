// Visits to the app on its public site, counted with GoatCounter
// (https://www.goatcounter.com): no cookies, nothing about the visitor. The
// page sends GoatCounter's counting request itself rather than loading its
// script, because the script sends the page's query string too, and the
// app's URL can hold private things: a link's `?dataset=` and `?model=`, a
// shared view's `#!` state with its volumes' addresses, and Hugging Face's
// sign-in code on the way back from it. What is sent: the fixed path below,
// a title, the screen's width and the referring site's origin. Never from
// another host (a local or development server, the tests), nor from a
// browser driven by automation, nor from one that has turned it off: as
// GoatCounter's script does, `#toggle-goatcounter` at the end of the
// address turns counting this browser's visits off, or on again.

/** David's GoatCounter site, which his other projects count on too: each
 * starts its paths with its own name. */
const CODE = "davidackerman"; // https://davidackerman.goatcounter.com
const LIVE = "janelia-cellmap.github.io";
const PATH = "/cellmap-flow/app/";
/** GoatCounter's script's own hash and key, so the toggle is the one its
 * documentation describes. */
const TOGGLE = "#toggle-goatcounter";
const SKIP_KEY = "skipgc";

/** Flips the setting when the address ends in TOGGLE, says which way it
 * now is, and takes the hash off before the app reads it as a view. */
function toggled() {
  if (location.hash !== TOGGLE) return;
  try {
    const off = localStorage.getItem(SKIP_KEY) !== "t";
    if (off) localStorage.setItem(SKIP_KEY, "t");
    else localStorage.removeItem(SKIP_KEY);
    alert(`Visits from this browser are ${off ? "no longer counted" : "counted again"} (GoatCounter).`);
  } catch {
    alert("This browser's storage is not available, so counting its visits cannot be turned off.");
  }
  const url = new URL(location.href);
  url.hash = "";
  history.replaceState(null, "", url);
}

function skipped() {
  try {
    return localStorage.getItem(SKIP_KEY) === "t";
  } catch {
    return false;
  }
}

export function countVisit() {
  toggled();
  if (location.hostname !== LIVE || navigator.webdriver || skipped()) return;
  let ref = "";
  try {
    ref = document.referrer ? new URL(document.referrer).origin : "";
  } catch {
    // an unparseable referrer is left out
  }
  const query = new URLSearchParams({
    p: PATH,
    t: "cellmap-flow: browser app",
    e: "false",
    s: String(screen.width),
    r: ref,
    rnd: Math.random().toString(36).slice(2, 7),
  });
  const url = `https://${CODE}.goatcounter.com/count?${query}`;
  // A beacon is a POST, which the service worker (src/sw.ts) leaves alone,
  // as it does the image an older browser falls back to.
  try {
    if (navigator.sendBeacon?.(url)) return;
  } catch {
    // falls back to the image
  }
  new Image().src = url;
}
