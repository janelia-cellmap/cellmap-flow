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
import { mkdtemp } from "node:fs/promises";
import { tmpdir } from "node:os";
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

/**
 * Build and serve the service worker as a single classic script.
 *
 * chunkmirage's Engine registers sw.js beside the page as a classic worker,
 * which cannot `import`. The page's build cannot produce it: as one entry
 * among the page's, Rolldown splits what the worker and the page share
 * (src/state/db.ts) into a chunk the worker would have to import. So the
 * worker is always bundled on its own, from src/sw.ts into one file with no
 * imports: in the build, into the output directory once the page is built;
 * in development, by a second Vite build in watch mode into a temporary
 * directory, served at /sw.js and rebuilt as its sources change (reload the
 * page to pick a new one up: the registration uses updateViaCache "none").
 */
export function serviceWorker(entry: string, publicPath = "/sw.js"): Plugin {
  let root = ".";
  let outDir = "dist";
  let building = false;
  return {
    name: "cellmap-flow:service-worker",
    configResolved(config) {
      root = config.root;
      outDir = resolve(config.root, config.build.outDir);
      building = config.command === "build";
    },
    async closeBundle() {
      if (!building) return;
      const { build } = await import("vite");
      await build({
        configFile: false,
        root,
        logLevel: "warn",
        publicDir: false,
        build: {
          outDir,
          emptyOutDir: false,
          copyPublicDir: false,
          modulePreload: false,
          rolldownOptions: {
            input: resolve(root, entry),
            output: { entryFileNames: "sw.js", format: "iife", inlineDynamicImports: true },
          },
        },
      });
    },
    async configureServer(server) {
      const { build } = await import("vite");
      const outDir = await mkdtemp(join(tmpdir(), "cellmap-flow-sw-"));
      const file = join(outDir, "sw.js");
      let failure: string | null = null;
      const watcher = await build({
        configFile: false,
        root: server.config.root,
        logLevel: "warn",
        publicDir: false,
        build: {
          outDir,
          emptyOutDir: true,
          copyPublicDir: false,
          modulePreload: false,
          watch: {},
          minify: false,
          sourcemap: "inline",
          rolldownOptions: {
            input: resolve(server.config.root, entry),
            output: { entryFileNames: "sw.js", format: "iife", inlineDynamicImports: true },
            onwarn(warning, warn) {
              warn(warning);
            },
          },
        },
      });
      if (watcher && typeof watcher === "object" && "on" in watcher) {
        const w = watcher as { on(event: string, handler: (e: { code: string; error?: Error }) => void): void; close?: () => Promise<void> };
        w.on("event", (event) => {
          if (event.code === "ERROR") {
            failure = event.error?.message ?? "unknown error";
            server.config.logger.error(`[sw] ${failure}`);
          } else if (event.code === "BUNDLE_END") {
            failure = null;
            server.config.logger.info("[sw] service worker rebuilt");
          }
        });
        server.httpServer?.once("close", () => void w.close?.());
      }
      server.middlewares.use((req, res, next) => {
        if ((req.url ?? "").split("?")[0] !== publicPath) return next();
        if (!existsSync(file)) {
          res.statusCode = 503;
          res.setHeader("Content-Type", "text/plain");
          res.end(failure ? `service worker build failed: ${failure}` : "service worker still building; retry");
          return;
        }
        res.setHeader("Content-Type", "text/javascript; charset=utf-8");
        res.setHeader("Cache-Control", "no-store");
        createReadStream(file).pipe(res);
      });
    },
  };
}
