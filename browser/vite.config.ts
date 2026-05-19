import { defineConfig, type Plugin } from "vite";
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

const here = path.dirname(fileURLToPath(import.meta.url));

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

// Neuroglancer gates bundled features by package.json "imports" conditions.
// These flip the :enabled variants on so zarr/n5/precomputed datasources,
// http/s3/etc kvstores, and image/segmentation/annotation layers get real
// code instead of the false-stub fallback.
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
    watch: {
      ignored: ["**/node_modules/**", "**/.vite/**", "**/dist/**", "**/public/**"],
      usePolling: true,
      interval: 1000,
    },
  },
  resolve: { conditions: NG_CONDITIONS },
  worker: { format: "es" },
  optimizeDeps: { include: ["neuroglancer"] },
  build: {
    rollupOptions: {
      input: {
        dashboard: path.resolve(here, "dashboard.html"),
        pipeline_builder: path.resolve(here, "pipeline_builder.html"),
      },
    },
  },
  plugins: [dashboardStatic()],
});
