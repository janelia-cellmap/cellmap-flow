// The browser build of cellmap-flow: the rendered dashboard (index.html, from
// build/render.py), its entry src/main.ts, and the service worker src/sw.ts,
// built for any base path (the Pages site serves it under /cellmap-flow/app/).
//
// The service worker keeps a fixed name, sw.js beside the page, because
// chunkmirage's Engine registers it by that name with scope ./ .
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
import { existsSync } from "node:fs";
import { resolve } from "node:path";
import type { ProxyOptions } from "vite";
import { defineConfig } from "vitest/config";

import { dashboardStatic, localNeuroglancer } from "./build/vite-plugins";

const here = import.meta.dirname;
const DASHBOARD_STATIC = resolve(here, "../cellmap_flow/dashboard/static");
const NEUROGLANCER_BUILD = resolve(here, "ng");
const NEUROGLANCER_DEMO = "https://neuroglancer-demo.appspot.com";

const ngProxy: Record<string, ProxyOptions> = existsSync(NEUROGLANCER_BUILD)
  ? {}
  : { "/ng": { target: NEUROGLANCER_DEMO, changeOrigin: true, rewrite: (p: string) => p.replace(/^\/ng/, "") } };

export default defineConfig({
  base: "./",
  publicDir: false,
  plugins: [dashboardStatic(DASHBOARD_STATIC, "static"), localNeuroglancer(NEUROGLANCER_BUILD, "ng")],
  build: {
    target: "es2022",
    chunkSizeWarningLimit: 1500, // onnxruntime-web and the zstd/blosc codecs (WebAssembly)
    rolldownOptions: {
      input: { index: resolve(here, "index.html"), sw: resolve(here, "src/sw.ts") },
      output: { entryFileNames: (chunk) => (chunk.name === "sw" ? "sw.js" : "assets/[name]-[hash].js") },
    },
  },
  worker: { format: "es" },
  // chunkmirage's Pyodide workers import its Python from web/chunkmirage/src as text
  server: { proxy: ngProxy, fs: { allow: [here, resolve(here, "chunkmirage"), DASHBOARD_STATIC] } },
  preview: { proxy: ngProxy },
  test: {
    environment: "node",
    include: ["test/**/*.test.ts"],
  },
});
