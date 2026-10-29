// Parity of web/src/shared/pasted.ts with cellmap_flow/portable/pasted.py: a
// dataset path as a person pastes it, from Neuroglancer with its format or as
// it is. The table (test/fixtures/pasted_paths.json) is written by hand, and
// tests/portable/test_pasted.py holds the Python to the same cases.
import { describe, expect, it } from "vitest";

import { pastedPath } from "../../src/shared/pasted";
import { loadFixture } from "./helpers";

const { cases } = loadFixture<{ cases: { pasted: string; path: string }[] }>("pasted_paths.json");

describe("a pasted dataset path reads as the path pasted_path gives", () => {
  for (const { pasted, path } of cases) {
    it(JSON.stringify(pasted), () => {
      expect(pastedPath(pasted)).toBe(path);
    });
  }
});
