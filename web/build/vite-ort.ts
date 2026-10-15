// Vite plugin that self-hosts onnxruntime-web's WebAssembly runtime.
//
// The ORT bundle the page imports ("onnxruntime-web/webgpu", see
// src/infer/session.ts) fetches its WebAssembly binary at run time from
// env.wasm.wasmPaths, which the page points at ./ort/ beside itself. This
// plugin serves node_modules/onnxruntime-web/dist's runtime files there in
// development and copies them into the build under the same prefix, so the
// deployed site carries the exact version npm installed (Marwan's prototype
// loaded them from a CDN, pinned by hand). Only the asyncify build is needed:
// it is the one the webgpu bundle names, and it runs the WASM provider too.
//
// chunkmirage ships the same plugin (web/chunkmirage/web/vite-ort.ts), but
// at the pinned commit it locates the package by resolving
// "onnxruntime-web/package.json", which the package's exports map does not
// expose: Node 22 throws ERR_PACKAGE_PATH_NOT_EXPORTED in configResolved.
// This one finds node_modules by path instead. Switch to chunkmirage's once
// a commit that resolves an exported file is pinned.
import { copyFileSync, createReadStream, existsSync, mkdirSync } from "node:fs";
import { extname, join, resolve } from "node:path";
import type { Plugin } from "vite";

const RUNTIME_FILES = ["ort-wasm-simd-threaded.asyncify.wasm", "ort-wasm-simd-threaded.asyncify.mjs"];
const MIME: Record<string, string> = { ".wasm": "application/wasm", ".mjs": "text/javascript; charset=utf-8" };

export function ortRuntime(prefix = "ort"): Plugin {
  const dist = resolve(import.meta.dirname, "../node_modules/onnxruntime-web/dist");
  let outDir = "dist";
  return {
    name: "cellmap-flow:ort-runtime",
    // The ORT bundle is one self-contained module; left out of the dev
    // server's dependency pre-bundling it is served as it is, and its own
    // import.meta.url and wasm loading are not rewritten.
    config: () => ({ optimizeDeps: { exclude: ["onnxruntime-web"] } }),
    configResolved(config) {
      outDir = resolve(config.root, config.build.outDir);
    },
    configureServer(server) {
      server.middlewares.use((req, res, next) => {
        const url = (req.url ?? "").split("?")[0];
        const file = RUNTIME_FILES.find((f) => url === `/${prefix}/${f}`);
        if (!file || !existsSync(join(dist, file))) return next();
        res.setHeader("Content-Type", MIME[extname(file)] ?? "application/octet-stream");
        createReadStream(join(dist, file)).pipe(res);
      });
    },
    generateBundle(_options, bundle) {
      // The ORT bundle also names its binary with new URL(..., import.meta.url)
      // as a fallback for when wasmPaths is unset, and Vite turns that into
      // a hashed copy under assets/. The page always sets wasmPaths, so that
      // copy is never fetched: drop it rather than ship 27 MB twice.
      for (const name of Object.keys(bundle)) if (/ort-wasm-simd-threaded.*\.wasm$/.test(name)) delete bundle[name];
    },
    closeBundle() {
      mkdirSync(join(outDir, prefix), { recursive: true });
      for (const file of RUNTIME_FILES) copyFileSync(join(dist, file), join(outDir, prefix, file));
    },
  };
}
