// The browser build of cellmap-flow: the rendered dashboard (index.html, from
// build/render.py), its entry src/main.ts, and the service worker src/sw.ts,
// built for any base path (the Pages site serves it under /cellmap-flow/app/).
//
// The service worker, sw.js beside the page (chunkmirage's Engine registers it
// by that name with scope ./), is bundled on its own by the serviceWorker
// plugin: a classic worker cannot import a chunk shared with the page.
//
// The dashboard's own JavaScript and CSS (cellmap_flow/dashboard/static) are
// not bundled: they are served and copied verbatim under ./static/, so the
// browser mode runs byte for byte the files the Flask dashboard serves. The
// entry loads the dashboard's main.js at runtime once the backend is ready.
//
// Neuroglancer must be same-origin (the service worker only sees requests from
// pages on its origin): in development /ng/ is relayed from appspot (upstream
// Neuroglancer, no painting) unless web/ng/ holds a build of the fork, which
// is then served instead; the deployed site always carries its own build.
import { existsSync, realpathSync } from "node:fs";
import { resolve } from "node:path";
import { defineConfig, type ProxyOptions } from "vite";

import { ortRuntime } from "./chunkmirage/web/vite-ort.ts";
import { dashboardStatic, localNeuroglancer, serviceWorker } from "./build/vite-plugins.ts";

const here = import.meta.dirname;
const DASHBOARD_STATIC = resolve(here, "../cellmap_flow/dashboard/static");
// The Python both modes run, bundled as text for the Pyodide workers (build/portable.ts).
const PORTABLE_PYTHON = resolve(here, "../cellmap_flow/portable");
const NEUROGLANCER_BUILD = resolve(here, "ng");
// The clone, or a local checkout linked in its place (CHUNKMIRAGE_LOCAL in
// scripts/fetch-chunkmirage.sh): Vite allows files by their real path.
const CHUNKMIRAGE = existsSync(resolve(here, "chunkmirage")) ? realpathSync(resolve(here, "chunkmirage")) : resolve(here, "chunkmirage");
const NEUROGLANCER_DEMO = "https://neuroglancer-demo.appspot.com";

const ngProxy: Record<string, ProxyOptions> = existsSync(NEUROGLANCER_BUILD)
  ? {}
  : { "/ng": { target: NEUROGLANCER_DEMO, changeOrigin: true, rewrite: (p: string) => p.replace(/^\/ng/, "") } };

export default defineConfig({
  base: "./",
  publicDir: false,
  plugins: [serviceWorker("src/sw.ts"), dashboardStatic(DASHBOARD_STATIC, "static"), localNeuroglancer(NEUROGLANCER_BUILD, "ng"), ortRuntime("ort")],
  build: {
    target: "es2022",
    chunkSizeWarningLimit: 1500, // onnxruntime-web and the zstd/blosc codecs (WebAssembly)
    rolldownOptions: {
      input: { index: resolve(here, "index.html") },
    },
  },
  worker: { format: "es" },
  // chunkmirage's Pyodide workers import its Python from web/chunkmirage/src as text, and ours from cellmap_flow/portable
  server: { proxy: ngProxy, fs: { allow: [here, CHUNKMIRAGE, DASHBOARD_STATIC, PORTABLE_PYTHON] } },
  preview: { proxy: ngProxy },
});
