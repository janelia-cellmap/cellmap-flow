// The shareable URL's pure parts (src/viewer/share.ts): the `#!` hash the
// page carries, round-tripped, and decoded as the Python neuroglancer
// package writes it.
import { describe, expect, it } from "vitest";

import { decodeState, encodeState } from "../../src/viewer/share";

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

describe("encodeState and decodeState", () => {
  it("round-trip a state", () => {
    const hash = encodeState(state);
    expect(hash.startsWith("#!")).toBe(true);
    expect(decodeState(hash)).toEqual(state);
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
});
