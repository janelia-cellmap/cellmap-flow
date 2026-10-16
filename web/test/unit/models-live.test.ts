// The catalog against the real Hugging Face repo. Opt in with
// CELLMAP_FLOW_LIVE=1: it needs the network, the Hub rate-limits anonymous
// requests, and it downloads one model file (the smallest, about 6 MB).
//
//   CELLMAP_FLOW_LIVE=1 npx vitest run test/unit/models-live
import { describe, expect, it } from "vitest";

import { createCatalog } from "../../src/models";
import { MemoryByteCache } from "../../src/models/byte-cache";
import { MemoryKeyValueStore } from "../../src/models/listing-cache";
import { modelInfo } from "../../src/shared/model-info";
import { fakeLog, FOLDERS } from "./models-helpers";

const live = process.env.CELLMAP_FLOW_LIVE === "1";

describe.skipIf(!live)("cellmap/onnx-web-models on the Hub", () => {
  it("lists the models, downloads the smallest file and describes it", { timeout: 120_000 }, async () => {
    const { log, lines } = fakeLog();
    const catalog = createCatalog({ log, storage: new MemoryKeyValueStore(), bytes: new MemoryByteCache() });

    const entries = await catalog.list();
    expect(entries.map((e) => e.folder)).toEqual(FOLDERS);
    expect(lines).toContain("model catalog: 8 models from Hugging Face");
    for (const entry of entries) {
      expect(entry.variant.variant).toBe("fp32");
      expect(modelInfo(entry).read_shape.map((n, i) => n - modelInfo(entry).write_shape[i]!)).toEqual([88, 88, 88]);
      expect(catalog.metadataFor(entry).description).toContain("ONNX fp32");
    }

    // setup_53's int8w file at its smallest block: 5.76 MB per the manifest.
    const setup53 = (await catalog.resolve("mito_aff_setup53_16nm"))!;
    const small = setup53.manifest.variants.find((v) => v.variant === "int8w" && v.input_size[0] === 124)!;
    const entry = { ...setup53, variant: small, onnxUrl: setup53.onnxUrl.replace(setup53.variant.file, small.file) };
    const progress: [number, number | null][] = [];
    const bytes = await catalog.modelBytes(entry, (loaded, total) => progress.push([loaded, total]));
    expect(bytes.byteLength).toBeGreaterThan(5_000_000);
    expect(bytes.byteLength / 1e6).toBeCloseTo(small.file_mb, 0);
    // The CDN sends Content-Length, so progress knows the total from the first chunk.
    expect(progress[0]![1]).toBe(bytes.byteLength);
    expect(progress.at(-1)).toEqual([bytes.byteLength, bytes.byteLength]);
    // An ONNX file is a protobuf whose first field is ir_version: tag 0x08.
    expect(new Uint8Array(bytes)[0]).toBe(0x08);
  });
});
