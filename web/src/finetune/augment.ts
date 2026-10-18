// Training-time augmentation of a whole-network tile or block, the port of
// the server's (cellmap_flow/finetune/data/augment.py): a random brightness
// (×0.8 to ×1.2) and Gaussian noise of 1% of the value range on the raw as
// stored, before it is normalized, then random flips of each axis and a
// random quarter turn in the y–x plane of the raw and the paint alike.
//
// The targets are made from the paint after it is turned, never turned
// themselves: an affinity's channels name directions (a flip makes the left
// neighbour the right one, a quarter turn makes y into x), so turning a
// target would need its channels permuted and shifted, while the paint's
// labels mean the same wherever they lie. The same goes for the starting
// network's logits that distillation holds the model to: they are worked
// out on the turned input, not turned. So the run turns the raw and the
// paint, then makes both from them (full/run-full.ts).
//
// A sample is a block of (z, y, x) in the order the network takes it: one
// slice for a 2D model (its slice axis first, as tiles.ts hands it over), a
// cube for a 3D model, whose flips along z are real ones. The raw and the
// paint share the block's centre: the raw is the paint's box grown by the
// same context on each side, so the same flip or turn of each about its own
// centre keeps them registered. z is flipped but never turned into, as on
// the server: EM is often coarser along z.
import type { Step } from "../contracts";

/** One tile's draw: a flip per axis (z, y, x), the quarter turns in y–x (0 to 3), the brightness factor. */
export interface AugmentDraw {
  flips: [boolean, boolean, boolean];
  turns: number;
  scale: number;
}

/** A draw from `random`: each axis flipped with even odds, 0 to 3 quarter
 * turns when the y–x plane may be turned (`turnable`: square, and its two
 * voxel sizes the same), and a brightness factor from 0.8 to 1.2. */
export function drawAugmentation(random: () => number, turnable: boolean): AugmentDraw {
  const flips: [boolean, boolean, boolean] = [random() < 0.5, random() < 0.5, random() < 0.5];
  const turns = turnable ? Math.min(3, Math.floor(random() * 4)) : 0;
  return { flips, turns, scale: 0.8 + 0.4 * random() };
}

type Values = Float32Array | Uint8Array | Uint16Array;

/** `data`, a block of `shape` (z, y, x, C order), flipped along each axis
 * `draw` flips, then turned `draw.turns` quarter turns in y–x (which needs
 * y and x the same length): a new array of the same type. */
export function transformBlock<T extends Values>(data: T, shape: readonly [number, number, number], draw: AugmentDraw): T {
  const [d, h, w] = shape;
  if (data.length !== d * h * w) throw new Error(`a block of ${shape.join("×")} has ${d * h * w} values, not ${data.length}`);
  const turns = ((draw.turns % 4) + 4) % 4;
  if (turns && h !== w) throw new Error(`a quarter turn needs a square y–x plane, not ${h}×${w}`);
  const [fz, fy, fx] = draw.flips;
  const out = new (data.constructor as new (n: number) => T)(data.length);
  for (let z = 0; z < d; z++) {
    const sz = fz ? d - 1 - z : z;
    for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) {
      // The voxel of the flipped block that lands at (y, x) after the turns.
      let a = y, b = x;
      for (let k = 0; k < turns; k++) { const t = a; a = b; b = h - 1 - t; }
      const sy = fy ? h - 1 - a : a, sx = fx ? w - 1 - b : b;
      out[(z * h + y) * w + x] = data[(sz * h + sy) * w + sx];
    }
  }
  return out;
}

const INTEGER_RANGES: Record<string, [number, number]> = {
  uint8: [0, 255], int8: [-128, 127], uint16: [0, 65535], int16: [-32768, 32767],
  uint32: [0, 4294967295], int32: [-2147483648, 2147483647],
};

/** The range the raw is kept inside (augment.py _intensity_range): the first
 * normalizer's window when it is a MinMaxNormalizer, the range the data is
 * known to use and is clipped to next anyway; else an integer dtype's own;
 * else none (a float raw, which keeps its negative half). */
export function intensityRange(norm: readonly Step[], dtype: string): [number, number] | null {
  const first = norm[0];
  if (first?.name === "MinMaxNormalizer") {
    const low = Number(first.min_value ?? 0), high = Number(first.max_value ?? 255);
    if (high > low) return [low, high];
  }
  return INTEGER_RANGES[dtype] ?? null;
}

/** A standard normal draw from `random` (Box–Muller). */
function gaussian(random: () => number): number {
  const u = 1 - random(), v = random();
  return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
}

/** The raw voxels `inside` the data scaled by `scale`, plus Gaussian noise of
 * 1% of `range` (of the block's largest magnitude when there is none),
 * clipped to `range`; 0 outside, where the model stage pads with zeros. */
export function augmentIntensity(raw: Float32Array, inside: Uint8Array, scale: number, range: [number, number] | null, random: () => number): Float32Array {
  let span: number;
  if (range) {
    span = range[1] - range[0];
  } else {
    let largest = 0;
    for (let i = 0; i < raw.length; i++) if (inside[i]) largest = Math.max(largest, Math.abs(raw[i]));
    span = Math.max(1, largest);
  }
  const sigma = 0.01 * span, out = new Float32Array(raw.length);
  for (let i = 0; i < raw.length; i++) {
    if (!inside[i]) continue;
    const v = raw[i] * scale + sigma * gaussian(random);
    out[i] = range ? Math.min(Math.max(v, range[0]), range[1]) : v;
  }
  return out;
}

/** What a run's draws were, for its log: proof that augmentation ran, and ran as drawn. */
export class AugmentTally {
  private n = 0;
  private flipped = [0, 0, 0];
  private turned = [0, 0, 0, 0];
  private low = Infinity;
  private high = -Infinity;

  add(draw: AugmentDraw): void {
    this.n++;
    draw.flips.forEach((f, a) => { if (f) this.flipped[a]++; });
    this.turned[draw.turns]++;
    this.low = Math.min(this.low, draw.scale);
    this.high = Math.max(this.high, draw.scale);
  }

  /** "120 tiles augmented: flipped along rows 49%, columns 52%; turned 0°: 26%, …; brightness ×0.80 to ×1.20".
   * `names` name the block's axes (z, y, x), null for one a tile is a voxel
   * thick along (a slice's z); `noun` what was drawn for, a tile or a block. */
  describe(names: readonly (string | null)[], turnable: boolean, noun = "tile"): string {
    if (!this.n) return `no ${noun}s augmented`;
    const pct = (k: number) => `${Math.round((100 * k) / this.n)}%`;
    const flips = names.map((name, a) => (name ? `${name} ${pct(this.flipped[a])}` : "")).filter(Boolean).join(", ");
    const turns = turnable ? `; turned ${this.turned.map((k, i) => `${i * 90}°: ${pct(k)}`).join(", ")}` : "";
    return `${this.n} ${noun}${this.n === 1 ? "" : "s"} augmented: flipped along ${flips}${turns}; brightness ×${this.low.toFixed(2)} to ×${this.high.toFixed(2)}`;
  }
}
