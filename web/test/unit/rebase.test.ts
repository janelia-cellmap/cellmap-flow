import { describe, expect, it } from "vitest";

import { rebaseApiUrl } from "../../src/rebase";

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
