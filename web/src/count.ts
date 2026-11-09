// Visits to the app on its public site, counted with GoatCounter
// (https://www.goatcounter.com): no cookies, nothing about the visitor. The
// page sends GoatCounter's counting request itself rather than loading its
// script, because the script sends the page's query string too, and the
// app's URL can hold private things: a link's `?dataset=` and `?model=`, a
// shared view's `#!` state with its volumes' addresses, and Hugging Face's
// sign-in code on the way back from it. What is sent: the fixed path below,
// a title, the screen's width and the referring site's origin. Never from
// another host (a local or development server, the tests), nor from a
// browser driven by automation.

/** David's GoatCounter site, which his other projects count on too: each
 * starts its paths with its own name. */
const CODE = "davidackerman"; // https://davidackerman.goatcounter.com
const LIVE = "janelia-cellmap.github.io";
const PATH = "/cellmap-flow/app/";

export function countVisit() {
  if (location.hostname !== LIVE || navigator.webdriver) return;
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
