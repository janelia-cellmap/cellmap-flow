// A box of a view served in chunks, assembled from them: what
// Inference.readOutput does with the chunks of a model's output view
// (engine.ts), apart from the engine so it runs in node. The chunks the box
// touches are asked for at once, each once, and each one's part of the box
// is copied in, row by row, every channel. A chunk is whole (chunkmirage's
// staged views place a block the volume ends in whole in its chunk, the rest
// fill), channels first: [C, chunk z, chunk y, chunk x], C order.

/** Voxels [lo, hi) (z, y, x) of a view chunked by `chunk`, with `channels`
 * channels, from `read(index)`, the chunk at grid index `index` as above:
 * [C, z, y, x] float32 in C order. */
export async function boxFromChunks(
  lo: readonly number[],
  hi: readonly number[],
  chunk: readonly number[],
  channels: number,
  read: (index: number[]) => Promise<Float32Array>,
): Promise<{ data: Float32Array; shape: number[] }> {
  const shape = hi.map((h, a) => h - lo[a]);
  if (shape.length !== 3 || chunk.length !== 3 || shape.some((n) => !(n > 0))) throw new Error(`the box ${lo.join(",")}..${hi.join(",")} is not a box of three axes`);
  const n = shape[0] * shape[1] * shape[2];
  const cn = chunk[0] * chunk[1] * chunk[2];
  const data = new Float32Array(channels * n);
  const first = lo.map((l, a) => Math.floor(l / chunk[a]));
  const last = hi.map((h, a) => Math.floor((h - 1) / chunk[a]));
  const reads: Promise<void>[] = [];
  for (let z = first[0]; z <= last[0]; z++) {
    for (let y = first[1]; y <= last[1]; y++) {
      for (let x = first[2]; x <= last[2]; x++) {
        const index = [z, y, x];
        reads.push(
          read(index).then((values) => {
            if (values.length !== channels * cn) throw new Error(`chunk ${index.join(",")} has ${values.length} values, not ${channels}×${chunk.join("×")}`);
            const cLo = index.map((i, a) => i * chunk[a]);
            const oLo = cLo.map((c, a) => Math.max(c, lo[a]));
            const oHi = cLo.map((c, a) => Math.min(c + chunk[a], hi[a]));
            const run = oHi[2] - oLo[2];
            for (let c = 0; c < channels; c++) {
              for (let zz = oLo[0]; zz < oHi[0]; zz++) {
                for (let yy = oLo[1]; yy < oHi[1]; yy++) {
                  const from = c * cn + ((zz - cLo[0]) * chunk[1] + (yy - cLo[1])) * chunk[2] + (oLo[2] - cLo[2]);
                  const to = c * n + ((zz - lo[0]) * shape[1] + (yy - lo[1])) * shape[2] + (oLo[2] - lo[2]);
                  data.set(values.subarray(from, from + run), to);
                }
              }
            }
          }),
        );
      }
    }
  }
  await Promise.all(reads);
  return { data, shape: [channels, ...shape] };
}
