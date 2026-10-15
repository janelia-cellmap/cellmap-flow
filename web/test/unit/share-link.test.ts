// The Share button's link (src/share-link.ts): the volume and the models in
// the query, the view in the hash, nothing without a volume.
import { describe, expect, it } from "vitest";

import { parseLink } from "../../src/link";
import { shareLink } from "../../src/share-link";
import { encodeState } from "../../src/viewer/share";

const PAGE = "https://janelia-cellmap.github.io/cellmap-flow/app/";
const DATASET = "https://janelia-cosem-datasets.s3.amazonaws.com/jrc_mus-liver/jrc_mus-liver.zarr/recon-1/em/fibsem-uint8";
const MODELS = [{ folder: "setup_53" }, { folder: "setup_54" }];

describe("shareLink", () => {
  it("is nothing without a volume", () => {
    expect(shareLink(PAGE, null, MODELS, null)).toBeNull();
  });

  it("names the volume and the models readably, and parseLink reads them back", () => {
    const link = shareLink(PAGE, DATASET, MODELS, null)!;
    expect(link).toBe(`${PAGE}?dataset=${DATASET}&model=setup_53&model=setup_54`);
    expect(parseLink(new URL(link).search)).toEqual({ datasetUrl: DATASET, models: ["setup_53", "setup_54"] });
  });

  it("carries the viewer's state hash, and drops the page's own query and hash", () => {
    const hash = encodeState({ position: [1, 2, 3], layers: [] });
    expect(shareLink(`${PAGE}?x=1#!old`, DATASET, [], `${PAGE}ng/index.html${hash}`)).toBe(`${PAGE}?dataset=${DATASET}${hash}`);
    expect(shareLink(PAGE, DATASET, [], `${PAGE}ng/index.html`)).toBe(`${PAGE}?dataset=${DATASET}`);
  });

  it("escapes what would end or split the query, and reads back the same", () => {
    const odd = "https://h/a b.zarr/x&y#z+w";
    const link = shareLink(PAGE, odd, [], null)!;
    expect(link).not.toMatch(/[ #+]/);
    expect(parseLink(new URL(link).search).datasetUrl).toBe(odd);
  });
});
