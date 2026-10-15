// Vite plugins for the browser build.
//
// dashboardStatic: serve a directory verbatim under a prefix in development,
// and copy it into the build under the same prefix. Used for the dashboard's
// own static files (cellmap_flow/dashboard/static), which are shared with the
// Flask app and must not be bundled or rewritten.
//
// localNeuroglancer: when web/ng/ holds a Neuroglancer build (the fork that
// paints, built by scripts/fetch-neuroglancer.sh), serve it under /ng/ in
// development and copy it into the build; otherwise do nothing, and
// vite.config.ts relays /ng/ from appspot instead.
import { cpSync, createReadStream, existsSync, statSync } from "node:fs";
import { extname, join, resolve } from "node:path";
import type { Connect, Plugin } from "vite";

const MIME: Record<string, string> = {
  ".html": "text/html; charset=utf-8",
  ".js": "text/javascript; charset=utf-8",
  ".mjs": "text/javascript; charset=utf-8",
  ".css": "text/css; charset=utf-8",
  ".json": "application/json; charset=utf-8",
  ".map": "application/json; charset=utf-8",
  ".wasm": "application/wasm",
  ".svg": "image/svg+xml",
  ".png": "image/png",
  ".ico": "image/x-icon",
  ".woff": "font/woff",
  ".woff2": "font/woff2",
};

/** A middleware serving the files of `dir` under `/<prefix>/`. */
function serveDir(dir: string, prefix: string): Connect.NextHandleFunction {
  const root = resolve(dir);
  return (req, res, next) => {
    const url = req.url ?? "";
    if (!url.startsWith(`/${prefix}/`)) return next();
    let rel = url.slice(prefix.length + 2).split("?")[0];
    try {
      rel = decodeURIComponent(rel);
    } catch {
      return next();
    }
    if (rel === "" || rel.endsWith("/")) rel += "index.html";
    const file = resolve(root, rel);
    if (!file.startsWith(root) || !existsSync(file) || statSync(file).isDirectory()) return next();
    res.setHeader("Content-Type", MIME[extname(file)] ?? "application/octet-stream");
    createReadStream(file).pipe(res);
  };
}

export function dashboardStatic(dir: string, prefix: string): Plugin {
  let outDir = "dist";
  return {
    name: "cellmap-flow:dashboard-static",
    configResolved(config) {
      outDir = resolve(config.root, config.build.outDir);
    },
    configureServer(server) {
      server.middlewares.use(serveDir(dir, prefix));
    },
    closeBundle() {
      cpSync(dir, join(outDir, prefix), { recursive: true });
    },
  };
}

export function localNeuroglancer(dir: string, prefix: string): Plugin {
  let outDir = "dist";
  return {
    name: "cellmap-flow:local-neuroglancer",
    configResolved(config) {
      outDir = resolve(config.root, config.build.outDir);
    },
    configureServer(server) {
      if (existsSync(dir)) server.middlewares.use(serveDir(dir, prefix));
    },
    closeBundle() {
      if (existsSync(dir)) cpSync(dir, join(outDir, prefix), { recursive: true });
    },
  };
}
