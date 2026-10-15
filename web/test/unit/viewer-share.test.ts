// The shareable URL's pure parts (src/viewer/share.ts): the `#!` hash the
// page carries, decoded from every form Neuroglancer writes or reads.
import { describe, expect, it } from "vitest";

import { decodeState, encodeState, isStateHash, urlSafeToJson, withHash } from "../../src/viewer/share";

const state = {
  dimensions: { z: [8e-9, "m"], y: [8e-9, "m"], x: [8e-9, "m"] },
  layers: [
    { type: "image", source: "zarr://https://x/dataset.zarr/recon-1/em/fibsem-uint8", shader: "void main(){}", name: "data" },
    { type: "image", source: "zarr3://http://h/m/", shader: "void main(){}", name: "model", visible: false },
  ],
  position: [10.5, 20.5, 30.5],
  layout: "xy",
  crossSectionScale: 2,
};

/** Neuroglancer's own writing of its hash (UrlHashBinding in the local
 * build): encodeURI of the JSON, with a few more characters escaped. */
function neuroglancerHash(json: object): string {
  const text = encodeURI(JSON.stringify(json)).replace(/[!'()*;,]/g, (c) => `%${c.charCodeAt(0).toString(16).toUpperCase()}`);
  return `#!${text}`;
}

describe("isStateHash", () => {
  it("is a #! hash with a state in it", () => {
    expect(isStateHash("#!%7B%7D")).toBe(true);
    expect(isStateHash("#!{'layers':[]}")).toBe(true);
  });
  it("is not an empty hash, a diff, or a state to fetch", () => {
    expect(isStateHash("")).toBe(false);
    expect(isStateHash("#")).toBe(false);
    expect(isStateHash("#!")).toBe(false);
    expect(isStateHash("#!+%7B%7D")).toBe(false);
    expect(isStateHash("#!https://example.org/state.json")).toBe(false);
    expect(isStateHash("#!gs://bucket/state.json")).toBe(false);
    expect(isStateHash("#tab=finetune")).toBe(false);
  });
});

describe("encodeState and decodeState", () => {
  it("round-trip a state", () => {
    const hash = encodeState(state);
    expect(hash.startsWith("#!")).toBe(true);
    expect(decodeState(hash)).toEqual(state);
  });
  it("read what Neuroglancer writes to its own hash", () => {
    expect(decodeState(neuroglancerHash(state))).toEqual(state);
  });
  it("read what the Python neuroglancer package writes (url_state.to_url_fragment)", () => {
    // Printed by neuroglancer.url_state.to_url_fragment for the state above
    // with gpuMemoryLimit set; the Python API wraps sources in a list.
    const fragment = "%7B%22gpuMemoryLimit%22:2000000000,%22dimensions%22:%7B%22z%22:%5B8e-09,%22m%22%5D,%22y%22:%5B8e-09,%22m%22%5D,%22x%22:%5B8e-09,%22m%22%5D%7D,"
      + "%22layers%22:%5B%7B%22type%22:%22image%22,%22source%22:%5B%7B%22url%22:%22zarr://https://x/dataset.zarr/recon-1/em/fibsem-uint8%22%7D%5D,%22shader%22:%22void%20main()%7B%7D%22,%22name%22:%22data%22%7D,"
      + "%7B%22type%22:%22image%22,%22source%22:%5B%7B%22url%22:%22zarr3://http://h/m/%22%7D%5D,%22shader%22:%22void%20main()%7B%7D%22,%22name%22:%22model%22,%22visible%22:false%7D%5D,"
      + "%22position%22:%5B10.5,20.5,30.5%5D,%22layout%22:%22xy%22,%22crossSectionScale%22:2.0%7D";
    const decoded = decodeState(`#!${fragment}`);
    expect(decoded).toEqual({
      gpuMemoryLimit: 2000000000,
      dimensions: state.dimensions,
      layers: [
        { type: "image", source: [{ url: "zarr://https://x/dataset.zarr/recon-1/em/fibsem-uint8" }], shader: "void main(){}", name: "data" },
        { type: "image", source: [{ url: "zarr3://http://h/m/" }], shader: "void main(){}", name: "model", visible: false },
      ],
      position: [10.5, 20.5, 30.5],
      layout: "xy",
      crossSectionScale: 2,
    });
  });
  it("read the oldest links' url-safe JSON: underscores for commas, single quotes", () => {
    const legacy = "#!{'layers':[{'type':'image'_'source':'zarr://https://x/a.zarr'_'name':'data'}]_'layout':'xy'_'position':[1_2_3]}";
    expect(decodeState(legacy)).toEqual({ layers: [{ type: "image", source: "zarr://https://x/a.zarr", name: "data" }], layout: "xy", position: [1, 2, 3] });
  });
  it("read a hash that is not percent-encoded, even with a stray percent sign", () => {
    expect(decodeState('#!{"layers":[],"note":"100%"}')).toEqual({ layers: [], note: "100%" });
  });
  it("are null for anything that is not a state", () => {
    expect(decodeState("")).toBeNull();
    expect(decodeState("#!")).toBeNull();
    expect(decodeState("#!+%7B%7D")).toBeNull();
    expect(decodeState("#!https://example.org/state.json")).toBeNull();
    expect(decodeState("#!not json at all")).toBeNull();
    expect(decodeState("#!%5B1%2C2%5D")).toBeNull(); // a list is not a state
  });
});

describe("urlSafeToJson", () => {
  it("leaves plain JSON alone, underscores in strings included", () => {
    const json = '{"a":[1,2],"b":"x_y, z","c":"it\'s"}';
    expect(urlSafeToJson(json)).toBe(json);
  });
  it("turns url-safe commas and single-quoted strings into JSON", () => {
    expect(urlSafeToJson("{'a':[1_2]_'b':'x_y'}")).toBe('{"a":[1,2],"b":"x_y"}');
    expect(urlSafeToJson("{'q':'say \\'hi\\' \"there\"'}")).toBe('{"q":"say \'hi\' \\"there\\""}');
    expect(urlSafeToJson("[1&2]")).toBe("[1,2]");
  });
});

describe("withHash", () => {
  it("replaces or adds the fragment", () => {
    expect(withHash("https://h/app/?x=1#!old", "#!new")).toBe("https://h/app/?x=1#!new");
    expect(withHash("https://h/app/", "#!new")).toBe("https://h/app/#!new");
  });
  it("drops the fragment for an empty hash", () => {
    expect(withHash("https://h/app/#!old", "")).toBe("https://h/app/");
  });
});
