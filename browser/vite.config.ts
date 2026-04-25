import { defineConfig, type Plugin } from "vite";
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

const here = path.dirname(fileURLToPath(import.meta.url));
const ortDir = path.resolve(here, "node_modules", "onnxruntime-web", "dist");

function shouldServe(name: string): boolean {
  return /(^ort-wasm.*\.(wasm|mjs)|^ort\.webgpu\..*\.mjs)$/.test(name);
}

const mime: Record<string, string> = {
  ".wasm": "application/wasm",
  ".mjs": "application/javascript",
  ".js": "application/javascript",
};

// Expose cellmap_flow's dashboard static dir at /dashboard-static/. The
// dashboard HTML (rendered by scripts/render-dashboard.py) references its
// CSS/JS/images via that prefix.
const dashboardStaticDir = path.resolve(here, "..", "cellmap_flow", "dashboard", "static");

function dashboardStatic(): Plugin {
  return {
    name: "cellmap-flow-dashboard-static",
    configureServer(server) {
      server.middlewares.use("/dashboard-static", (req, res, next) => {
        if (!req.url) return next();
        const rel = decodeURIComponent(req.url.split("?")[0].replace(/^\/+/, ""));
        const p = path.join(dashboardStaticDir, rel);
        if (!p.startsWith(dashboardStaticDir) || !fs.existsSync(p) || fs.statSync(p).isDirectory()) {
          return next();
        }
        const ext = path.extname(p);
        const types: Record<string, string> = {
          ".js": "application/javascript",
          ".css": "text/css",
          ".png": "image/png",
          ".jpg": "image/jpeg",
          ".jpeg": "image/jpeg",
          ".svg": "image/svg+xml",
          ".ico": "image/x-icon",
          ".woff": "font/woff",
          ".woff2": "font/woff2",
        };
        res.setHeader("Content-Type", types[ext] ?? "application/octet-stream");
        fs.createReadStream(p).pipe(res);
      });
    },
    generateBundle() {
      if (!fs.existsSync(dashboardStaticDir)) return;
      const walk = (dir: string, prefix: string) => {
        for (const entry of fs.readdirSync(dir, { withFileTypes: true })) {
          const abs = path.join(dir, entry.name);
          const rel = prefix ? `${prefix}/${entry.name}` : entry.name;
          if (entry.isDirectory()) walk(abs, rel);
          else {
            this.emitFile({
              type: "asset",
              fileName: `dashboard-static/${rel}`,
              source: fs.readFileSync(abs),
            });
          }
        }
      };
      walk(dashboardStaticDir, "");
    },
  };
}

function ortAssets(): Plugin {
  return {
    name: "cellmap-flow-ort-assets",
    configureServer(server) {
      server.middlewares.use("/ort", (req, res, next) => {
        if (!req.url) return next();
        const name = decodeURIComponent(req.url.split("?")[0].replace(/^\/+/, ""));
        if (!shouldServe(name)) return next();
        const p = path.join(ortDir, name);
        if (!p.startsWith(ortDir) || !fs.existsSync(p)) return next();
        res.setHeader("Content-Type", mime[path.extname(p)] ?? "application/octet-stream");
        res.setHeader("Cross-Origin-Resource-Policy", "same-origin");
        fs.createReadStream(p).pipe(res);
      });
    },
    generateBundle() {
      for (const f of fs.readdirSync(ortDir)) {
        if (!shouldServe(f)) continue;
        this.emitFile({
          type: "asset",
          fileName: `ort/${f}`,
          source: fs.readFileSync(path.join(ortDir, f)),
        });
      }
    },
  };
}

// Neuroglancer gates bundled features by package.json "imports" conditions.
// These flip the :enabled variants on so zarr/n5/precomputed datasources,
// http/s3/etc kvstores, and image/segmentation/annotation layers get real
// code instead of the false-stub fallback. (See tourguide/web-app/vite.config.ts.)
const NG_CONDITIONS = [
  "neuroglancer/datasource/zarr:enabled",
  "neuroglancer/datasource/n5:enabled",
  "neuroglancer/datasource/precomputed:enabled",
  "neuroglancer/kvstore/http:enabled",
  "neuroglancer/kvstore/s3:enabled",
  "neuroglancer/kvstore/gcs:enabled",
  "neuroglancer/kvstore/gzip:enabled",
  "neuroglancer/kvstore/byte_range:enabled",
  "neuroglancer/kvstore/zip:enabled",
  "neuroglancer/layer/image:enabled",
  "neuroglancer/layer/segmentation:enabled",
  "neuroglancer/layer/annotation:enabled",
];

export default defineConfig({
  server: {
    headers: {
      "Cross-Origin-Opener-Policy": "same-origin",
      "Cross-Origin-Embedder-Policy": "credentialless",
    },
    watch: {
      ignored: ["**/node_modules/**", "**/.vite/**", "**/dist/**", "**/public/**"],
      usePolling: true,
      interval: 1000,
    },
  },
  resolve: {
    conditions: NG_CONDITIONS,
  },
  worker: {
    format: "es",
  },
  optimizeDeps: {
    exclude: ["onnxruntime-web"],
    include: ["neuroglancer"],
  },
  plugins: [ortAssets()],
});
