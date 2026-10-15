// Shared by the models-*.test.ts files: the real manifests and tree listings
// saved from cellmap/onnx-web-models on 2026-10-14 (fixtures/onnx-web-models),
// and a stand-in for fetch that answers as the Hub does, so the catalog can
// be exercised without a network.
import { readFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

import type { GpuLimits, ModelEntry } from "../../src/contracts";
import type { FetchLike } from "../../src/models/hf";
import { entryFor, parseManifest } from "../../src/models/manifest";
import { chooseVariant } from "../../src/models/variant";

const FIXTURES = join(dirname(fileURLToPath(import.meta.url)), "fixtures", "onnx-web-models");

/** The folders the saved listing has (every model folder of the repo). */
export const FOLDERS = ["setup_53", "setup_54", "setup_55", "setup_56", "setup_61", "setup_67"];

export function readFixture(name: string): unknown {
  return JSON.parse(readFileSync(join(FIXTURES, name), "utf8"));
}

/** `folder`'s manifest.json as fetched (unknown: not yet parsed). */
export function rawManifest(folder: string): unknown {
  return readFixture(`${folder}.manifest.json`);
}

/** The repo's top-level tree listing (the folders, conv3d/ among them, README, bench). */
export function treeListing(): unknown {
  return readFixture("tree.main.json");
}

/** A ModelEntry for `folder`: with `limits`, the variant chooseVariant picks
 * for them; without, the fp32 middle block (132³ for a ResUNet, 140³ for a
 * Unet3dIso), a fixed point for the tests' numbers whatever the chooser's
 * default for unknown limits is. */
export function entryFixture(folder = "setup_54", limits?: GpuLimits): ModelEntry {
  const manifest = parseManifest(rawManifest(folder), `${folder}/manifest.json`);
  if (limits) return entryFor(folder, manifest, chooseVariant(manifest, limits));
  const fp32 = manifest.variants.filter((v) => v.variant === "fp32").sort((a, b) => a.input_size[0]! - b.input_size[0]!);
  return entryFor(folder, manifest, fp32[Math.floor((fp32.length - 1) / 2)]!);
}

export function jsonResponse(data: unknown, status = 200): Response {
  return new Response(JSON.stringify(data), { status, headers: { "content-type": "application/json" } });
}

export function statusResponse(status: number, body = ""): Response {
  return new Response(body, { status });
}

export interface HubOptions {
  /** The status to answer the tree listing with, instead of 200 and the fixture. */
  treeStatus?: number;
  /** Folders whose manifest answers with this status instead of the fixture. */
  manifestStatus?: Record<string, number>;
  /** Bytes to answer a .onnx URL with (by file name), streamed in chunks of `chunkSize`. */
  onnx?: Record<string, Uint8Array>;
  onnxStatus?: number;
  chunkSize?: number;
  /** Leave out Content-Length on the .onnx answer. */
  noContentLength?: boolean;
}

export interface FakeHub {
  fetch: FetchLike;
  /** Every URL asked for, in order. */
  calls: string[];
}

/** A fetch that answers like the Hub, from the fixtures. */
export function fakeHub(options: HubOptions = {}): FakeHub {
  const calls: string[] = [];
  const fetch: FetchLike = async (url) => {
    calls.push(url);
    if (/\/api\/models\/[^/]+\/[^/]+\/tree\/main$/.test(url)) {
      return options.treeStatus && options.treeStatus !== 200 ? statusResponse(options.treeStatus) : jsonResponse(treeListing());
    }
    const manifest = /\/raw\/main\/([^/]+)\/manifest\.json$/.exec(url);
    if (manifest) {
      const folder = manifest[1]!;
      const status = options.manifestStatus?.[folder];
      if (status && status !== 200) return statusResponse(status);
      if (!FOLDERS.includes(folder)) return statusResponse(404, "Entry not found");
      return jsonResponse(rawManifest(folder));
    }
    const onnx = /\/resolve\/main\/[^/]+\/([^/]+\.onnx)$/.exec(url);
    if (onnx) {
      if (options.onnxStatus && options.onnxStatus !== 200) return statusResponse(options.onnxStatus);
      const bytes = options.onnx?.[onnx[1]!];
      if (!bytes) return statusResponse(404, "Entry not found");
      return streamedResponse(bytes, options.chunkSize ?? 7, !options.noContentLength);
    }
    return statusResponse(404, `unexpected URL ${url}`);
  };
  return { fetch, calls };
}

/** `bytes` as a response whose body streams in `chunkSize` pieces. */
export function streamedResponse(bytes: Uint8Array, chunkSize: number, withLength: boolean): Response {
  let offset = 0;
  const body = new ReadableStream<Uint8Array>({
    pull(controller) {
      if (offset >= bytes.byteLength) {
        controller.close();
        return;
      }
      controller.enqueue(bytes.slice(offset, offset + chunkSize));
      offset += chunkSize;
    },
  });
  const headers: Record<string, string> = { "content-type": "application/octet-stream" };
  if (withLength) headers["content-length"] = String(bytes.byteLength);
  return new Response(body, { status: 200, headers });
}

/** A log that keeps its lines. */
export function fakeLog(): { log: (line: string) => void; lines: string[] } {
  const lines: string[] = [];
  return { log: (line) => lines.push(line), lines };
}
