// Connected components of one channel's nonzero voxels: the browser side of
// cellmap_flow/post/postprocessors.py's LabelPostprocessor, which is
// cellmap_flow/post/segment.py's connected_components (scipy.ndimage.label
// with generate_binary_structure(3, connectivity)), remove_small,
// as_connectivity and as_bool.
//
// Ids are numbered as scipy numbers them: by the raster (C) order in which
// each object's first voxel is met, 1 for the object met first. Checked
// against scipy on hundreds of random 2-D and 3-D masks at every
// connectivity, so the parity test compares ids exactly. The labelling is
// the classic two passes with a union-find over provisional labels: the
// first pass gives each foreground voxel the label of a foreground
// neighbour already visited (joining them when they differ) or a new one,
// the second resolves each provisional label to its object's final id. An
// object's first voxel in raster order has no visited foreground neighbour,
// so it gets a new provisional label, and that label is the smallest in the
// object; numbering the objects by their smallest provisional label, in
// order, is numbering them by their first voxel, as scipy does.
//
// remove_small is affinity.ts's removeSmall, the same segment.py function
// (survivors numbered 1..n again in the order of their ids); it returns
// uint64, which the labels come back from as uint32.
import type { StepValue } from "../contracts";
import { removeSmall } from "./affinity";
import type { Declares, NdArray, NdData, PostprocessorDef } from "./post";

// ---------------------------------------------------------------------------
// Parsing the step's parameters as the Python constructors do
// ---------------------------------------------------------------------------

/** Python's int() of a form value: an integer (a string one may carry sign
 * and whitespace, "4" but not "4.5"), a number truncated toward zero, a
 * bool as 0 or 1. An absent or empty value is `fallback` (the dashboard's
 * empty field means the default). */
export function asInt(value: StepValue | undefined, fallback: number, what: string): number {
  if (value === undefined || value === null || value === "") return fallback;
  if (typeof value === "boolean") return value ? 1 : 0;
  if (typeof value === "number") {
    if (!Number.isFinite(value)) throw new Error(`${what} must be an integer, got ${JSON.stringify(value)}`);
    return Math.trunc(value);
  }
  if (!/^\s*[+-]?\d+\s*$/.test(value)) throw new Error(`${what} must be an integer, got ${JSON.stringify(value)}`);
  return Number.parseInt(value, 10);
}

/** segment.as_connectivity: `value` as scipy's connectivity for an `ndim`-D
 * array, 1 (faces), 2 (faces and edges) or 3 (all neighbours), capped at
 * `ndim` (3 asks for every neighbour, which in a 2-D slice is 2). Absent:
 * 1, the Python default. */
export function asConnectivity(value: StepValue | undefined, ndim = 3): number {
  let connectivity: number;
  try {
    connectivity = asInt(value, 1, "connectivity");
  } catch {
    throw new Error(`connectivity must be 1, 2 or 3, got ${JSON.stringify(value)}`);
  }
  if (![1, 2, 3].includes(connectivity)) throw new Error(`connectivity must be 1, 2 or 3, got ${JSON.stringify(value)}`);
  return Math.min(connectivity, ndim);
}

/** segment.as_bool: a form's string is true when it is "true", "1", "yes"
 * or "on" (any case, trimmed); anything else by Python's truth. Absent:
 * false, the Python default. */
export function asBoolLike(value: StepValue | undefined): boolean {
  if (value === undefined || value === null) return false;
  if (typeof value === "string") return ["true", "1", "yes", "on"].includes(value.trim().toLowerCase());
  return Boolean(value);
}

// ---------------------------------------------------------------------------
// Union-find over dense integer labels
// ---------------------------------------------------------------------------

/** A union-find over the integers 0..n-1, in typed arrays that grow as
 * elements are added: union by size, path halving on find. */
export class UnionFind {
  private parent: Uint32Array;
  private size: Uint32Array;
  private n = 0;

  constructor(capacity = 64) {
    this.parent = new Uint32Array(Math.max(1, capacity));
    this.size = new Uint32Array(Math.max(1, capacity));
  }

  /** How many elements there are. */
  get count(): number { return this.n; }

  /** A new element in a set of its own; returns its index. */
  add(): number {
    if (this.n === this.parent.length) {
      const parent = new Uint32Array(this.parent.length * 2);
      parent.set(this.parent);
      this.parent = parent;
      const size = new Uint32Array(this.size.length * 2);
      size.set(this.size);
      this.size = size;
    }
    const i = this.n++;
    this.parent[i] = i;
    this.size[i] = 1;
    return i;
  }

  /** The representative of `a`'s set. */
  find(a: number): number {
    const parent = this.parent;
    while (parent[a] !== a) {
      parent[a] = parent[parent[a]];  // path halving: point at the grandparent
      a = parent[a];
    }
    return a;
  }

  /** Join the sets of `a` and `b`; true when they were apart. */
  union(a: number, b: number): boolean {
    let ra = this.find(a), rb = this.find(b);
    if (ra === rb) return false;
    if (this.size[ra] < this.size[rb]) [ra, rb] = [rb, ra];
    this.parent[rb] = ra;
    this.size[ra] += this.size[rb];
    return true;
  }
}

// ---------------------------------------------------------------------------
// Labelling
// ---------------------------------------------------------------------------

/** The neighbour offsets [dz, dy, dx] of scipy's generate_binary_structure
 * (`ndim`, `connectivity`) that come BEFORE a voxel in raster order: those a
 * one-pass scan has already visited. For 2-D (a z plane) dz is 0. A
 * neighbour touches when its offsets sum to at most `connectivity` in
 * absolute value: 1 gives the 3 face neighbours, 2 adds 6 edge neighbours,
 * 3 the 4 corner ones (13 in all); in 2-D, 2 and 4. */
export function earlierOffsets(ndim: 2 | 3, connectivity: number): [number, number, number][] {
  const offsets: [number, number, number][] = [];
  for (const dz of ndim === 3 ? [-1, 0, 1] : [0]) {
    for (const dy of [-1, 0, 1]) {
      for (const dx of [-1, 0, 1]) {
        const touches = Math.abs(dz) + Math.abs(dy) + Math.abs(dx);
        if (touches === 0 || touches > connectivity) continue;
        const earlier = dz < 0 || (dz === 0 && (dy < 0 || (dy === 0 && dx < 0)));
        if (earlier) offsets.push([dz, dy, dx]);
      }
    }
  }
  return offsets;
}

/** The objects of a (z, y, x) foreground mask (nonzero = foreground), each
 * voxel's id in a uint32 array, 0 for background; `count` is how many.
 * `earlier` says which visited neighbours touch (earlierOffsets). */
export function labelComponents(mask: Uint8Array, shape: readonly number[], earlier: readonly (readonly number[])[]): { labels: Uint32Array; count: number } {
  const [nz, ny, nx] = shape;
  const n = nz * ny * nx;
  if (mask.length !== n) throw new Error(`a mask of ${mask.length} voxels is not of shape ${shape.join("×")}`);
  // Provisional labels, written over with the final ids in the second pass.
  const labels = new Uint32Array(n);
  const sets = new UnionFind(1024);
  sets.add();  // element 0 stands for the background, so labels index the elements directly
  const offsets = earlier.flat();
  const m = earlier.length;
  let i = 0;
  for (let z = 0; z < nz; z++) {
    for (let y = 0; y < ny; y++) {
      for (let x = 0; x < nx; x++, i++) {
        if (!mask[i]) continue;
        let label = 0;
        for (let k = 0; k < m; k++) {
          const zz = z + offsets[3 * k], yy = y + offsets[3 * k + 1], xx = x + offsets[3 * k + 2];
          if (zz < 0 || yy < 0 || yy >= ny || xx < 0 || xx >= nx) continue;  // zz < nz always, as dz <= 0
          const other = labels[(zz * ny + yy) * nx + xx];
          if (!other) continue;
          if (!label) label = other;
          else if (other !== label) sets.union(label, other);
        }
        labels[i] = label || sets.add();
      }
    }
  }
  // Final ids by the smallest provisional label of each object: the
  // provisional labels were handed out in raster order of first voxels.
  const ids = new Uint32Array(sets.count);
  let count = 0;
  for (let p = 1; p < sets.count; p++) {
    const root = sets.find(p);
    if (!ids[root]) ids[root] = ++count;
    ids[p] = ids[root];
  }
  for (let k = 0; k < n; k++) labels[k] = ids[labels[k]];
  return { labels, count };
}

/** `data` as a foreground mask: 1 where nonzero (mask != 0). */
export function foregroundOf(data: NdData): Uint8Array {
  const mask = new Uint8Array(data.length);
  if (data instanceof BigUint64Array) {
    for (let i = 0; i < data.length; i++) mask[i] = data[i] !== 0n ? 1 : 0;
  } else {
    for (let i = 0; i < data.length; i++) mask[i] = data[i] !== 0 ? 1 : 0;  // NaN counts as foreground, as numpy's != 0 does
  }
  return mask;
}

export interface ComponentOptions {
  /** 1, 2 or 3 (as asConnectivity gives it). */
  connectivity: number;
  /** Objects of fewer voxels become background. */
  minSize: number;
  /** Each z plane labelled on its own, in 2-D, its ids following the planes before. */
  perSlice: boolean;
}

/** segment.connected_components: an id per object of the nonzero voxels of
 * a (z, y, x) block, as uint32. */
export function connectedComponents(data: NdData, shape: readonly number[], options: ComponentOptions): Uint32Array {
  if (shape.length !== 3) throw new Error(`connected components label (z, y, x) blocks, got shape ${shape.join("×")}`);
  const mask = foregroundOf(data);
  const [nz, ny, nx] = shape;
  let labels: Uint32Array;
  if (options.perSlice) {
    const earlier = earlierOffsets(2, Math.min(options.connectivity, 2));
    const plane = ny * nx;
    labels = new Uint32Array(mask.length);
    let count = 0;
    for (let z = 0; z < nz; z++) {
      const slice = labelComponents(mask.subarray(z * plane, (z + 1) * plane), [1, ny, nx], earlier);
      for (let i = 0; i < plane; i++) if (slice.labels[i]) labels[z * plane + i] = slice.labels[i] + count;
      count += slice.count;
    }
  } else {
    labels = labelComponents(mask, shape, earlierOffsets(3, options.connectivity)).labels;
  }
  // remove_small returns at once for a min_size of 1 or less; not calling
  // it then spares the default chain a uint64 round trip on every chunk.
  return options.minSize > 1 ? toUint32(removeSmall(labels, options.minSize)) : labels;
}

/** `data` as uint32, as numpy's astype(np.uint32) casts on x86: truncation
 * toward zero, modulo 2^32 (what storing into a Uint32Array does). */
export function toUint32(data: NdData): Uint32Array {
  const out = new Uint32Array(data.length);
  if (data instanceof BigUint64Array) {
    for (let i = 0; i < data.length; i++) out[i] = Number(BigInt.asUintN(32, data[i]));
  } else {
    for (let i = 0; i < data.length; i++) out[i] = data[i];
  }
  return out;
}

// ---------------------------------------------------------------------------
// The step
// ---------------------------------------------------------------------------

/** LabelPostprocessor(channel=0, connectivity=1, min_size=0, per_slice=False):
 * the whole block as uint32 with `channel` replaced by the labels of its
 * nonzero voxels; the other channels pass through, cast. */
export const labelPostprocessor: PostprocessorDef = {
  declares: (): Declares => ({ dtype: "uint32", isSegmentation: true }),
  make: (step) => {
    const channel = asInt(step.channel, 0, "LabelPostprocessor.channel");
    const options: ComponentOptions = {
      connectivity: asConnectivity(step.connectivity, 3),
      minSize: asInt(step.min_size, 0, "LabelPostprocessor.min_size"),
      perSlice: asBoolLike(step.per_slice),
    };
    return ({ data, shape }: NdArray): NdArray => {
      if (shape.length !== 4) throw new Error(`LabelPostprocessor expects a (c, z, y, x) block, got shape ${shape.join("×")}`);
      const [c, z, y, x] = shape;
      const ch = channel < 0 ? c + channel : channel;  // numpy indexing: -1 is the last channel
      if (ch < 0 || ch >= c) throw new Error(`LabelPostprocessor: channel ${channel} of an output with ${c} channel(s)`);
      const plane = z * y * x;
      const out = toUint32(data);
      const labels = connectedComponents(data.subarray(ch * plane, (ch + 1) * plane) as NdData, [z, y, x], options);
      out.set(labels, ch * plane);
      return { data: out, shape };
    };
  },
};
