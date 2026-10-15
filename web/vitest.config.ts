// vitest's configuration, separate from vite.config.ts: vitest bundles its own
// copy of Vite's types, so a `test` key inside vite.config.ts does not typecheck
// against Vite's own defineConfig. The unit tests need no plugins or proxies,
// so this does not merge the Vite config. Vitest picks this file by name.
import { defineConfig } from "vitest/config";

export default defineConfig({
  test: {
    environment: "node",
    include: ["test/**/*.test.ts"],
    exclude: ["test/e2e/**", "node_modules/**"],
  },
});
