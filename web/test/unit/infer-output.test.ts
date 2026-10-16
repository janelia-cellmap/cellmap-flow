// How Inference.readOutput assembles a box of a model's output from the
// chunks of its output view (infer/output.ts): each chunk the box touches
// asked for once, every voxel placed, a chunk of the wrong size refused.
import { describe, expect, it } from "vitest";

import { boxFromChunks } from "../../src/infer/output";

describe("boxFromChunks", () => {
  /** A [C, 10, 9, 8] array chunked by [4, 4, 4], chunks whole (past the end, -1). */
  const shape = [10, 9, 8], chunk = [4, 4, 4], C = 2;
  const value = (c: number, z: number, y: number, x: number) => c * 1000 + z * 100 + y * 10 + x;
  function chunkAt(index: number[]): Float32Array {
    const out = new Float32Array(C * 64);
    let i = 0;
    for (let c = 0; c < C; c++) for (let z = 0; z < 4; z++) for (let y = 0; y < 4; y++) for (let x = 0; x < 4; x++, i++) {
      const at = [index[0] * 4 + z, index[1] * 4 + y, index[2] * 4 + x];
      out[i] = at.every((v, a) => v < shape[a]) ? value(c, at[0], at[1], at[2]) : -1;
    }
    return out;
  }

  it("asks for each chunk the box touches once, and places every voxel", async () => {
    const asked: string[] = [];
    const lo = [1, 3, 2], hi = [10, 9, 7];
    const box = await boxFromChunks(lo, hi, chunk, C, async (index) => {
      asked.push(index.join(","));
      return chunkAt(index);
    });
    expect(box.shape).toEqual([C, 9, 6, 5]);
    expect(asked.sort()).toEqual(["0,0,0", "0,0,1", "0,1,0", "0,1,1", "0,2,0", "0,2,1", "1,0,0", "1,0,1", "1,1,0", "1,1,1", "1,2,0", "1,2,1", "2,0,0", "2,0,1", "2,1,0", "2,1,1", "2,2,0", "2,2,1"]);
    let i = 0;
    for (let c = 0; c < C; c++) for (let z = lo[0]; z < hi[0]; z++) for (let y = lo[1]; y < hi[1]; y++) for (let x = lo[2]; x < hi[2]; x++, i++) {
      expect(box.data[i]).toBe(value(c, z, y, x));
    }
  });

  it("refuses a chunk of the wrong size", async () => {
    await expect(boxFromChunks([0, 0, 0], [2, 2, 2], chunk, C, async () => new Float32Array(3))).rejects.toThrow("chunk 0,0,0 has 3 values, not 2×4×4×4");
  });
});
