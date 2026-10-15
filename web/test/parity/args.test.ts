// Parity of web/src/shared/args.ts with cellmap_flow/pipeline_spec.py and
// cellmap_flow/serving/protocol.py: the args blob in a layer's URL.
//
// A link shared between the server and browser modes works only if both
// write the same blob for the same chain and read each other's, so the blob
// must match byte for byte and the digest must match. The fixture
// (test/fixtures/args.json, from test/fixtures/make_fixtures.py) holds what
// PipelineSpec wrote for each chain. Two cases are marked `known_difference`
// because Python's json and JSON.stringify differ on them (an integral float
// renders as 0.0 against 0; non-ASCII is \u-escaped against UTF-8): for
// those only the decoded meaning is checked, and the fixture records why.
import { describe, expect, it } from "vitest";

import { ARGS_KEY, type Chain } from "../../src/contracts";
import { chainDigest, decodeArgs, encodeArgs, viewName } from "../../src/shared/args";
import { loadFixture } from "./helpers";

interface ArgsCase {
  id: string;
  /** The chain as PipelineSpec received it: a list of steps, or the legacy {Name: {params}} dict. */
  given: { input_norm: unknown; postprocess: unknown };
  /** As PipelineSpec normalizes it (to_json_data()), the form the browser works with. */
  chain: Chain;
  extras: Record<string, unknown>;
  blob: string;
  blob_without_extras: string;
  digest: string;
  digest_text: string;
  layer_url: string;
  known_difference: string | null;
}

interface ArgsFixture {
  args_key: string;
  dashboard_url: string;
  host: string;
  model: string;
  cases: ArgsCase[];
}

const fixture = loadFixture<ArgsFixture>("args.json");

describe("the args blob codec matches PipelineSpec", () => {
  it("uses the same marker", () => {
    expect(ARGS_KEY).toBe(fixture.args_key);
  });

  for (const c of fixture.cases) {
    describe(c.id, () => {
      it("decodes the Python blob to the chain and its extras", () => {
        expect(decodeArgs(c.blob)).toEqual({ chain: c.chain, extras: c.extras });
        expect(decodeArgs(c.blob_without_extras)).toEqual({ chain: c.chain, extras: {} });
      });

      if (c.known_difference === null) {
        it("encodes the chain to the same blob, byte for byte", () => {
          expect(encodeArgs(c.chain, c.extras)).toBe(c.blob);
          expect(encodeArgs(c.chain)).toBe(c.blob_without_extras);
        });

        it("computes the same digest", async () => {
          expect(c.extras.digest, "the fixture's digest extra is its digest").toBe(c.digest);
          expect(await chainDigest(c.chain)).toBe(c.digest);
        });
      } else {
        // Known difference (c.known_difference): the blobs need not be equal,
        // but the browser's own blob must still mean the same chain, and the
        // Python's blob must still decode here (checked above).
        it(`round-trips its own blob (known difference: ${c.known_difference})`, async () => {
          const blob = encodeArgs(c.chain, c.extras);
          expect(decodeArgs(blob)).toEqual({ chain: c.chain, extras: c.extras });
          expect(await chainDigest(c.chain)).toMatch(/^[0-9a-f]{16}$/);
        });
      }

      it("names the engine view as the Python names the layer's dataset", () => {
        const view = viewName(fixture.model, c.blob);
        expect(view).toBe(`${fixture.model}${ARGS_KEY}${c.blob}${ARGS_KEY}`);
        // viewer.layers.prediction_url(host, model, blob, engine="chunkmirage")
        expect(c.layer_url).toBe(`zarr://${fixture.host}/${view}/zarr`);
      });
    });
  }

  it("covers the legacy dict form through PipelineSpec's normalization", () => {
    // The browser never sees the {Name: {params}} form (PipelineSpec reads
    // it on the server side); what it must agree with is the normalized
    // list, with "name" first in each step.
    const legacy = fixture.cases.find((c) => c.id === "legacy_dict")!;
    expect(Array.isArray(legacy.given.input_norm)).toBe(false);
    for (const step of [...legacy.chain.input_norm, ...legacy.chain.postprocess]) {
      expect(Object.keys(step)[0]).toBe("name");
    }
    expect(encodeArgs(legacy.chain, legacy.extras)).toBe(legacy.blob);
  });
});
