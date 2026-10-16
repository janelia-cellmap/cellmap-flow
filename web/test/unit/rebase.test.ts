import { describe, expect, it } from "vitest";

import { apiDownloadUrl, dispositionFilename, rebaseApiUrl } from "../../src/rebase";

describe("rebaseApiUrl", () => {
  const pages = new URL("https://janelia-cellmap.github.io/cellmap-flow/app/");
  const root = new URL("http://localhost:5173/");

  it("moves a root-relative /api/ URL under the page's directory", () => {
    expect(rebaseApiUrl("/api/pipeline", pages)).toBe("https://janelia-cellmap.github.io/cellmap-flow/app/api/pipeline");
    expect(rebaseApiUrl("/api/finetune/logs/stream?offset=12", pages)).toBe(
      "https://janelia-cellmap.github.io/cellmap-flow/app/api/finetune/logs/stream?offset=12",
    );
  });

  it("is the identity at an origin's root, where the dashboard's URLs already work", () => {
    expect(rebaseApiUrl("/api/pipeline", root)).toBe("http://localhost:5173/api/pipeline");
  });

  it("leaves every other URL alone", () => {
    for (const url of ["./static/js/x.js", "https://huggingface.co/api/models", "api/pipeline", "/static/css/dark.css"]) {
      expect(rebaseApiUrl(url, pages)).toBe(url);
    }
  });
});

describe("apiDownloadUrl", () => {
  const pages = new URL("https://janelia-cellmap.github.io/cellmap-flow/app/");
  const page = "https://janelia-cellmap.github.io/cellmap-flow/app/?dataset=x";
  const root = new URL("http://localhost:5173/");

  it("takes a root-relative /api/ link, rebased, and one already under the page's directory as it is", () => {
    expect(apiDownloadUrl("/api/export-config", page, pages)).toBe("https://janelia-cellmap.github.io/cellmap-flow/app/api/export-config");
    // the Export Config link after installApiRebase rewrote it
    expect(apiDownloadUrl("https://janelia-cellmap.github.io/cellmap-flow/app/api/export-config", page, pages)).toBe(
      "https://janelia-cellmap.github.io/cellmap-flow/app/api/export-config",
    );
    // the Finetune tab's links, set after the rebase ran
    expect(apiDownloadUrl("/api/finetune/model/setup_53_2d-ft1/download", page, pages)).toBe(
      "https://janelia-cellmap.github.io/cellmap-flow/app/api/finetune/model/setup_53_2d-ft1/download",
    );
    expect(apiDownloadUrl("/api/finetune/volume/v1/download?x=1", "http://localhost:5173/", root)).toBe("http://localhost:5173/api/finetune/volume/v1/download?x=1");
  });

  it("leaves other links to the browser: another origin, another path, a blob: URL", () => {
    for (const href of ["https://huggingface.co/api/models", "./static/css/dark.css", "/static/x.yaml", "blob:https://janelia-cellmap.github.io/1234-5678"]) {
      expect(apiDownloadUrl(href, page, pages)).toBeNull();
    }
  });
});

describe("dispositionFilename", () => {
  it("reads the names the backend gives, quoted or not", () => {
    expect(dispositionFilename("attachment; filename=cellmap_flow_config_20261016_135727.yaml")).toBe("cellmap_flow_config_20261016_135727.yaml");
    expect(dispositionFilename('attachment; filename="v1.zarr.zip"')).toBe("v1.zarr.zip");
  });

  it("prefers an RFC 6266 filename*, decoded", () => {
    expect(dispositionFilename("attachment; filename=\"fallback.onnx\"; filename*=UTF-8''mito%20ft%C3%A9.onnx")).toBe("mito fté.onnx");
  });

  it("is null without a name", () => {
    expect(dispositionFilename(null)).toBeNull();
    expect(dispositionFilename("attachment")).toBeNull();
    expect(dispositionFilename('attachment; filename=""')).toBeNull();
  });
});
