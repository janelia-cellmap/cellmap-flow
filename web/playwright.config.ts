// The end-to-end smoke test (test/e2e/smoke.spec.ts): headless Chromium
// driving the real page against a Vite server that Playwright starts.
//
// Opt-in: `E2E=1 npm run e2e`. The test needs the network (a public dataset
// on S3, a model from Hugging Face) and, since headless Chromium has no
// WebGPU, runs inference on the WASM provider, which takes minutes. Without
// E2E=1 no server is started and the test reports itself skipped, so a plain
// `npm run e2e` is quick and harmless.
//
// Knobs (environment): E2E_PORT (default 4173); CHROME_PATH (a Chromium binary);
// E2E_WEBGPU=1 to run the models on this machine's GPU through WebGPU (see
// WEBGPU_ARGS) instead of on its CPU through WebAssembly;
// E2E_CHUNK_TIMEOUT_MS (how long to wait for the first chunk); E2E_SERVER=preview to serve
// the build in web/dist with `vite preview` (CI, after `npm run build`)
// instead of the dev server; E2E_DATASET and E2E_MODEL (see the test).
import { defineConfig, devices } from "@playwright/test";

const E2E = process.env.E2E === "1";
const PORT = Number(process.env.E2E_PORT ?? 4173);
// Literal localhost: the page's WebGPU and service worker need a secure
// context, and localhost is one without TLS.
const BASE_URL = `http://localhost:${PORT}/`;

// Chromium's flags for WebGPU on a Linux machine's GPU, headless. Chromium
// leaves WebGPU off on Linux (enable-unsafe-webgpu turns it on) and runs it
// over Vulkan (the Vulkan feature, and ANGLE on Vulkan); without a display,
// Vulkan's window surface cannot be made and Chromium falls back to
// SwiftShader, its CPU emulation of a GPU, unless the surface is left out.
// Found on a workstation with two RTX 2080 Ti and NVIDIA's Vulkan driver
// (535): with all four the page's adapter is the NVIDIA card (Turing, 4 GB
// buffers); without --use-angle=vulkan or --disable-vulkan-surface it is
// SwiftShader. A machine without a Vulkan driver for its GPU gets
// SwiftShader whatever the flags. macOS and Windows need none of this.
const WEBGPU_ARGS = ["--enable-unsafe-webgpu", "--enable-features=Vulkan", "--use-angle=vulkan", "--disable-vulkan-surface"];
const SERVER_COMMAND =
  process.env.E2E_SERVER === "preview"
    ? `npx vite preview --port ${PORT} --strictPort`
    : `npx vite --port ${PORT} --strictPort`;

export default defineConfig({
  testDir: "./test/e2e",
  // One slow test: a chunk on WASM can take several minutes.
  timeout: 15 * 60_000,
  expect: { timeout: 30_000 },
  fullyParallel: false,
  workers: 1,
  retries: 0,
  reporter: process.env.CI ? [["list"], ["html", { open: "never" }]] : "list",
  use: {
    baseURL: BASE_URL,
    trace: "retain-on-failure",
    screenshot: "only-on-failure",
    launchOptions: {
      // Containers (CI) give /dev/shm too little room for Chromium.
      args: ["--disable-dev-shm-usage", ...(process.env.E2E_WEBGPU === "1" ? WEBGPU_ARGS : [])],
      // CHROME_PATH: a Chromium of your own (e.g. one already in Playwright's
      // cache) when `npx playwright install chromium` cannot download one.
      ...(process.env.CHROME_PATH ? { executablePath: process.env.CHROME_PATH } : {}),
    },
  },
  projects: [{ name: "chromium", use: { ...devices["Desktop Chrome"] } }],
  webServer: E2E
    ? {
        command: SERVER_COMMAND,
        url: BASE_URL,
        reuseExistingServer: !process.env.CI,
        timeout: 120_000,
        stdout: "pipe",
        stderr: "pipe",
      }
    : undefined,
});
