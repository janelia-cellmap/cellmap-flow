// Objects from affinities by mutex watershed: the browser side of
// cellmap_flow/post/postprocessors.py's AffinityPostprocessor and of the two
// cellmap_flow/post/segment.py functions it runs, mutex_watershed and
// remove_small.
//
// The model outputs one channel per offset in `neighborhood`: channel c at
// voxel u is the affinity between u and u + offset_c, high inside an object,
// low across a boundary. The mutex watershed (Wolf et al., "The Mutex
// Watershed: Efficient, Parameter-Free Image Partitioning", ECCV 2018) turns
// them into objects in one greedy pass: every edge (u, v) with both ends
// inside the block has weight affinity - bias; the edges are taken in
// descending order of |weight|; a positive edge joins the two clusters
// unless a mutex already keeps them apart, a negative edge adds a mutex
// between the two clusters unless they are already one. The strongest
// evidence, for or against, is applied first and is never overruled.
//
// The Python calls mwatershed.agglom (mwatershed 0.5.3/0.5.4, a Rust
// crate), and agglom() below does what it does, checked against it in
// test/parity/affinity.test.ts. Its particulars, read from its source
// (src/lib.rs get_edges and agglomerate, src/clustering.rs):
//   - a voxel with no positive edge at all (every affinity it takes part in
//     is at or below the bias) is background, label 0, whatever the mutexes
//     did: it never merged and it never could;
//   - every other voxel is labelled with its cluster's root node's flat
//     index, and node 0's index is 0. So the voxel at the block's first
//     corner, when it stays a singleton (its positive edges all blocked by
//     mutexes), is background as well. A cluster of two or more is never
//     labelled 0: mwatershed's union-find puts the lower-numbered root under
//     the higher one when their ranks tie, so node 0 roots nothing but itself;
//   - an edge of weight exactly 0 is a split there (`aff > 0` is its merge
//     test). Here it does nothing: 0 is the smallest |weight|, so such edges
//     come last, after every edge that could merge, and a mutex added then
//     changes no cluster. With float affinities a weight of exactly 0 is a
//     coincidence in any case;
//   - edges of equal |weight| are taken in an order its unstable sort
//     decides. Ties are a coincidence with float affinities too, and the
//     parity fixtures are chosen without any.
// Then, as segment.mutex_watershed does: a fragment whose mean of the
// per-voxel mean affinity (over the channels whose offset pairs voxels in
// the block, of the affinities before the bias is subtracted) is below the
// bias is background too, since every voxel ends up in some fragment and
// this is where the model sees no object; the rest are numbered 1..n in
// order of first appearance in C order (fastremap.renumber); and
// remove_small drops fragments under min_size (the postprocessor passes
// none). Finally the ids are offset by the chunk's Morton increment
// (morton.ts), so they are unique across chunks, and returned as one uint64
// channel.
import type { Step } from "../contracts";
import { mortonIncrement } from "./morton";
import { asNumber } from "./norms";
import type { NdArray, NdData, PostprocessorDef } from "./post";

/** The Python constructor's default neighborhood: the offsets of the
 * cellmap affinity models' nine channels, as a Python list literal. */
/** The Python constructor's default `bias`: the midpoint of a sigmoid's
 * probabilities (with 0 every edge joined, and one object came out). */
export const DEFAULT_BIAS = 0.5;

export const DEFAULT_NEIGHBORHOOD = "[[1, 0, 0], [0, 1, 0], [0, 0, 1], [3, 0, 0], [0, 3, 0], [0, 0, 3], [9, 0, 0], [0, 9, 0], [0, 0, 9]]";

// --- the neighborhood parameter -------------------------------------------

/** `neighborhood` as the Python constructor reads it (ast.literal_eval of a
 * list literal): a list of offsets, each three integers (z, y, x). A string
 * is the dashboard's or the class default's Python literal, lists or tuples,
 * with the whitespace and trailing commas Python allows; a list (from a
 * YAML) is taken as it is. */
export function parseNeighborhood(value: unknown): number[][] {
  const given = value === undefined || value === null || value === "" ? DEFAULT_NEIGHBORHOOD : value;
  const parsed = typeof given === "string" ? parsePythonList(given) : given;
  if (!Array.isArray(parsed) || parsed.length === 0) {
    throw new Error(`AffinityPostprocessor.neighborhood: a list of offsets is needed, got ${JSON.stringify(value)}`);
  }
  return parsed.map((offset) => {
    if (!Array.isArray(offset) || offset.length !== 3 || !offset.every((n) => Number.isInteger(n))) {
      throw new Error(`AffinityPostprocessor.neighborhood: each offset is three integers (z, y, x), got ${JSON.stringify(offset)}`);
    }
    return offset.map(Number);
  });
}

/** A Python literal of nested lists or tuples of integers, such as
 * "[[1, 0, 0], (0, 1, 0),]": the part of ast.literal_eval a neighborhood
 * needs. Recursive descent over the text. */
function parsePythonList(text: string): unknown {
  let i = 0;
  const skipSpace = () => { while (i < text.length && /\s/.test(text[i]!)) i++; };
  const fail = (why: string): never => {
    throw new Error(`AffinityPostprocessor.neighborhood: ${why} at character ${i} of ${JSON.stringify(text)}`);
  };
  const value = (): unknown => {
    skipSpace();
    const open = text[i];
    if (open === "[" || open === "(") {
      const close = open === "[" ? "]" : ")";
      i++;
      const items: unknown[] = [];
      for (;;) {
        skipSpace();
        if (i >= text.length) fail(`missing ${close}`);
        if (text[i] === close) { i++; return items; }  // also after a trailing comma
        items.push(value());
        skipSpace();
        if (i >= text.length) fail(`missing ${close}`);
        if (text[i] === ",") { i++; continue; }
        if (text[i] === close) { i++; return items; }
        fail(`expected "," or "${close}"`);
      }
    }
    const integer = /^[+-]?\d+/.exec(text.slice(i));
    if (!integer) fail("expected an integer or a list");
    i += integer![0].length;
    return Number(integer![0]);
  };
  const result = value();
  skipSpace();
  if (i !== text.length) fail("unexpected text");
  return result;
}

// --- the graph ---------------------------------------------------------------

/** The edges of a block's affinity graph, flat C-order voxel indices: edge k
 * joins u[k] to v[k] = u[k] + offset with weight[k] = affinity - bias. */
interface Edges {
  count: number;
  u: Int32Array;
  v: Int32Array;
  weight: Float64Array;
}

/** Every edge of `aff` ([C, z, y, x], float64) for the first offsets.length
 * channels, with both ends inside the block (mwatershed's get_edges).
 * `hasPositive` is set at both ends of every edge with weight > 0: the
 * voxels mwatershed does not file as background. NaN is refused, as
 * mwatershed refuses it. */
function buildEdges(aff: Float64Array, spatial: number[], offsets: number[][], bias: number, hasPositive: Uint8Array): Edges {
  const [Z, Y, X] = spatial as [number, number, number];
  const N = Z * Y * X;
  // One pass to size the arrays exactly: a channel's edges are the voxels
  // whose partner is in bounds along every axis.
  let count = 0;
  for (const [oz, oy, ox] of offsets) {
    count += Math.max(0, Z - Math.abs(oz)) * Math.max(0, Y - Math.abs(oy)) * Math.max(0, X - Math.abs(ox));
  }
  const u = new Int32Array(count), v = new Int32Array(count), weight = new Float64Array(count);
  let k = 0;
  offsets.forEach(([oz, oy, ox], c) => {
    const stride = (oz * Y + oy) * X + ox;  // from u's flat index to v's
    const z0 = Math.max(0, -oz), z1 = Math.min(Z, Z - oz);
    const y0 = Math.max(0, -oy), y1 = Math.min(Y, Y - oy);
    const x0 = Math.max(0, -ox), x1 = Math.min(X, X - ox);
    const plane = c * N;
    for (let z = z0; z < z1; z++) {
      for (let y = y0; y < y1; y++) {
        const row = (z * Y + y) * X;
        for (let x = x0; x < x1; x++) {
          const from = row + x;
          const w = aff[plane + from]! - bias;
          if (Number.isNaN(w)) throw new Error("AffinityPostprocessor: the affinities contain NaN");
          u[k] = from;
          v[k] = from + stride;
          weight[k] = w;
          if (w > 0) { hasPositive[from] = 1; hasPositive[from + stride] = 1; }
          k++;
        }
      }
    }
  });
  return { count, u, v, weight };
}

const LITTLE_ENDIAN = new Uint8Array(new Uint16Array([1]).buffer)[0] === 1;

/** The edges' indices in descending order of |weight|: an LSD radix sort on
 * the weights' float64 bit patterns, four passes of 16 bits. The bits of a
 * non-negative double, read as an unsigned integer, order as its value does,
 * so the sign bit is masked off and the rest is sorted as a 64-bit key.
 * Exact, and about three times faster in node than a comparator sort of the
 * 650 k indices a 3-channel 60³ block has (the 9-channel default has 1.8 M).
 * The two 32-bit words of each double are read through a Uint32Array view,
 * which puts the low word first on a little-endian machine (every machine a
 * browser runs on; checked, since a big-endian one would sort wrongly). */
function orderByDescendingMagnitude(weight: Float64Array, count: number): Uint32Array {
  if (!LITTLE_ENDIAN) throw new Error("AffinityPostprocessor: this machine is big-endian; the radix sort reads little-endian doubles");
  const words = new Uint32Array(weight.buffer, weight.byteOffset, 2 * count);
  let order = new Uint32Array(count);
  for (let i = 0; i < count; i++) order[i] = i;
  let scratch = new Uint32Array(count);
  const starts = new Uint32Array(65536);
  for (let pass = 0; pass < 4; pass++) {
    const word = pass >> 1;  // the low word (bits 0-31) for passes 0 and 1, the high word for 2 and 3
    const shift = (pass & 1) * 16;
    const mask = word === 1 && shift === 16 ? 0x7fff : 0xffff;  // the top pass drops the sign bit
    const digit = (e: number) => (words[2 * e + word]! >>> shift) & mask;
    starts.fill(0);
    for (let i = 0; i < count; i++) starts[digit(order[i]!)]++;
    // Descending: the highest digit's run begins at 0.
    let position = 0;
    for (let d = 65535; d >= 0; d--) { const n = starts[d]!; starts[d] = position; position += n; }
    for (let i = 0; i < count; i++) { const e = order[i]!; scratch[starts[digit(e)]!++] = e; }
    [order, scratch] = [scratch, order];
  }
  return order;
}

/** A set of unordered pairs of cluster roots: which clusters a mutex keeps
 * apart. An open-addressing hash table in two typed arrays (linear probing,
 * grown at 60% full), since a Set per root cost most of the segmenter's
 * time: a 60³ block adds some hundred thousand mutexes. */
class RootPairs {
  private lo: Int32Array;  // the lower root of each slot's pair, -1 for an empty slot
  private hi: Int32Array;
  private mask: number;
  private count = 0;

  constructor(expected: number) {
    let slots = 1 << 12;
    while (slots * 0.6 < expected) slots *= 2;
    this.lo = new Int32Array(slots).fill(-1);
    this.hi = new Int32Array(slots);
    this.mask = slots - 1;
  }

  private static hash(lo: number, hi: number): number {
    let h = Math.imul(lo, 0x9e3779b1) ^ Math.imul(hi, 0x85ebca77);
    h ^= h >>> 15;
    h = Math.imul(h, 0xc2b2ae35);
    return (h ^ (h >>> 13)) >>> 0;
  }

  /** The slot holding (lo, hi), or the empty slot where it would go. */
  private slot(lo: number, hi: number): number {
    let s = RootPairs.hash(lo, hi) & this.mask;
    while (this.lo[s] !== -1 && !(this.lo[s] === lo && this.hi[s] === hi)) s = (s + 1) & this.mask;
    return s;
  }

  has(a: number, b: number): boolean {
    return this.lo[this.slot(Math.min(a, b), Math.max(a, b))] !== -1;
  }

  /** Adds the pair; false when it was there already. */
  add(a: number, b: number): boolean {
    const lo = Math.min(a, b), hi = Math.max(a, b);
    const s = this.slot(lo, hi);
    if (this.lo[s] !== -1) return false;
    this.lo[s] = lo;
    this.hi[s] = hi;
    if (++this.count > this.lo.length * 0.6) this.grow();
    return true;
  }

  private grow(): void {
    const lo = this.lo, hi = this.hi;
    this.lo = new Int32Array(lo.length * 2).fill(-1);
    this.hi = new Int32Array(lo.length * 2);
    this.mask = this.lo.length - 1;
    for (let i = 0; i < lo.length; i++) {
      if (lo[i] === -1) continue;
      const s = this.slot(lo[i]!, hi[i]!);
      this.lo[s] = lo[i]!;
      this.hi[s] = hi[i]!;
    }
  }
}

/** Union-find over the voxels with the mutexes between clusters: the mutex
 * watershed's state (mwatershed's Clustering, with its Positives and
 * Negatives), in typed arrays.
 *
 * The mutexes are kept twice: as a set of root pairs (RootPairs), which
 * answers "are these two clusters kept apart?", and as a list of partners
 * per root, which says what to re-key when the root merges into another:
 * each pair (m, other) becomes (m, root). The lists are linked lists in
 * typed arrays, append-only, and a merge joins the other's list onto the
 * root's in O(1). An entry's partner may be stale, a node that has since
 * merged into another root; find() resolves it when the entry is walked,
 * so the other root's list need not be touched when a partner merges. The
 * root with the longer list survives a merge, so each entry is walked
 * O(log n) times in all. */
class Clustering {
  private readonly parent: Int32Array;
  readonly size: Int32Array;
  private readonly pairs: RootPairs;
  // the partner lists: per root the first and last entry (-1 for none) and
  // the length; per entry the partner and the next entry
  private readonly head: Int32Array;
  private readonly tail: Int32Array;
  private readonly length: Int32Array;
  private partner = new Int32Array(1 << 16);
  private next = new Int32Array(1 << 16);
  private entries = 0;

  constructor(n: number, expectedMutexes: number) {
    this.parent = new Int32Array(n);
    for (let i = 0; i < n; i++) this.parent[i] = i;
    this.size = new Int32Array(n).fill(1);
    this.pairs = new RootPairs(expectedMutexes);
    this.head = new Int32Array(n).fill(-1);
    this.tail = new Int32Array(n).fill(-1);
    this.length = new Int32Array(n);
  }

  /** The root of `i`'s cluster, halving the path on the way. */
  find(i: number): number {
    while (this.parent[i] !== i) {
      const grandparent = this.parent[this.parent[i]!]!;
      this.parent[i] = grandparent;
      i = grandparent;
    }
    return i;
  }

  /** Whether a mutex keeps roots `a` and `b` apart. */
  apart(a: number, b: number): boolean {
    return this.pairs.has(a, b);
  }

  /** A mutex between roots `a` and `b` (nothing if there is one already). */
  split(a: number, b: number): void {
    if (!this.pairs.add(a, b)) return;
    this.append(a, b);
    this.append(b, a);
  }

  /** Roots `a` and `b` into one cluster, which inherits both sets of mutexes. */
  merge(a: number, b: number): void {
    const [root, other] = this.length[a]! >= this.length[b]! ? [a, b] : [b, a];
    this.parent[other] = root;
    this.size[root] = this.size[root]! + this.size[other]!;
    if (this.head[other] === -1) return;
    // every mutex of the other cluster is now a mutex of the merged one
    for (let e = this.head[other]!; e !== -1; e = this.next[e]!) {
      const m = this.find(this.partner[e]!);
      if (m !== root) this.pairs.add(m, root);
    }
    // ... and its partner list joins the root's, for the root's own merge later
    if (this.tail[root] === -1) this.head[root] = this.head[other]!;
    else this.next[this.tail[root]!] = this.head[other]!;
    this.tail[root] = this.tail[other]!;
    this.length[root] = this.length[root]! + this.length[other]!;
    this.head[other] = this.tail[other] = -1;
    this.length[other] = 0;
  }

  private append(root: number, partner: number): void {
    if (this.entries === this.partner.length) {
      const grownPartner = new Int32Array(this.entries * 2), grownNext = new Int32Array(this.entries * 2);
      grownPartner.set(this.partner);
      grownNext.set(this.next);
      this.partner = grownPartner;
      this.next = grownNext;
    }
    const e = this.entries++;
    this.partner[e] = partner;
    this.next[e] = -1;
    if (this.tail[root] === -1) this.head[root] = e;
    else this.next[this.tail[root]!] = e;
    this.tail[root] = e;
    this.length[root]!++;
  }
}

/** mwatershed.agglom(affinities - bias, offsets) on a block of affinities
 * `aff` ([C, z, y, x] as float64, C order; `spatial` is [z, y, x]), using
 * channel c for offsets[c]: a cluster id per voxel, nonzero for the voxels
 * in a cluster and 0 for the ones mwatershed files as background (see the
 * top of this file). The nonzero ids are arbitrary; mutexWatershed()
 * renumbers them. */
export function agglom(aff: Float64Array, spatial: number[], offsets: number[][], bias: number): Uint32Array {
  const N = spatial.reduce((a, b) => a * b, 1);
  const hasPositive = new Uint8Array(N);
  const edges = buildEdges(aff, spatial, offsets, bias, hasPositive);
  const order = orderByDescendingMagnitude(edges.weight, edges.count);
  const clusters = new Clustering(N, edges.count >> 2);  // a starting size for the mutex table; it grows
  for (let k = 0; k < edges.count; k++) {
    const e = order[k]!;
    const w = edges.weight[e]!;
    if (w === 0) break;  // the rest are zeros too: nothing left that can merge
    const u = edges.u[e]!, v = edges.v[e]!;
    // As mwatershed does, an edge to a voxel without any positive edge is
    // dropped: that voxel never merges, so a mutex on it would block nothing.
    if (!hasPositive[u] || !hasPositive[v]) continue;
    const a = clusters.find(u), b = clusters.find(v);
    if (a === b) continue;  // one cluster already
    if (w > 0) {
      if (!clusters.apart(a, b)) clusters.merge(a, b);  // unless kept apart already
    } else {
      clusters.split(a, b);
    }
  }
  // mwatershed labels a voxel with its root's flat index, so node 0 as a
  // root is label 0, background: it roots only itself (see the top).
  const originAlone = clusters.size[clusters.find(0)] === 1;
  const labels = new Uint32Array(N);
  for (let i = 0; i < N; i++) {
    if (!hasPositive[i] || (i === 0 && originAlone)) continue;  // background, 0
    labels[i] = clusters.find(i) + 1;
  }
  return labels;
}

// --- segment.py --------------------------------------------------------------

/** segment.remove_small(labels, min_size): `labels` (0 for background, ids
 * numbered without large gaps, as every segmenter's are) with objects of
 * fewer than `min_size` voxels made background and the rest numbered 1..n
 * again in the order of their ids, as uint64. A min_size of 1 or less
 * removes nothing. */
export function removeSmall(labels: ArrayLike<number> | BigUint64Array, minSize = 0): BigUint64Array {
  const out = new BigUint64Array(labels.length);
  const id = (i: number): number => Number(labels[i]);
  let maxId = 0;
  for (let i = 0; i < labels.length; i++) maxId = Math.max(maxId, id(i));
  const threshold = Math.trunc(minSize);  // int(min_size)
  if (!(threshold > 1) || maxId === 0) {
    for (let i = 0; i < labels.length; i++) out[i] = BigInt(id(i));
    return out;
  }
  const sizes = new Uint32Array(maxId + 1);
  for (let i = 0; i < labels.length; i++) sizes[id(i)]++;
  const newIds = new BigUint64Array(maxId + 1);  // 0 for the dropped and for background
  let next = 1n;
  for (let k = 1; k <= maxId; k++) {
    if (sizes[k]! >= threshold) newIds[k] = next++;
  }
  for (let i = 0; i < labels.length; i++) out[i] = newIds[id(i)]!;
  return out;
}

/** `data` as float64, as np.asarray(affinities, dtype=np.float64) reads a
 * block: the values as they are (uint64 ids through Number). */
function asFloat64(data: NdData): Float64Array {
  if (data instanceof Float64Array) return data;
  const out = new Float64Array(data.length);
  for (let i = 0; i < data.length; i++) out[i] = Number(data[i]);
  return out;
}

/** segment.mutex_watershed(affinities, offsets, bias, min_size): the
 * objects of a block of affinities `nd` ([C, z, y, x], in [0, 1]; taken as
 * float64, scaled by nobody here), as uint64 labels 1..n with no id skipped
 * and 0 for background, shape [z, y, x]. `offsets` beyond the channels are
 * cut, as the Python cuts them. */
export function mutexWatershed(nd: NdArray, offsets: number[][], bias: number, minSize = 0): BigUint64Array {
  const [C, ...spatial] = nd.shape as [number, number, number, number];
  const N = spatial.reduce((a, b) => a * b, 1);
  const aff = asFloat64(nd.data);
  const labels = agglom(aff, spatial, offsets.slice(0, C), bias);

  // A channel whose offset is as long as the block along an axis, or
  // longer, pairs no voxel in it (a 2D model's z channel on the one slice it
  // serves): agglom builds it no edge, and it is left out of the mean too,
  // as segment.mutex_watershed leaves it out. Its values are the model's
  // guess at a neighbour the block does not have, and they pulled a slice's
  // objects a third of the way to background. Every channel is kept when
  // none pairs a voxel.
  const unpaired = offsets.slice(0, C).map((offset) => offset.some((o, a) => Math.abs(o) >= spatial[a]!));
  const kept = Array.from({ length: C }, (_, c) => c).filter((c) => !unpaired[c]);
  const channels = kept.length ? kept : Array.from({ length: C }, (_, c) => c);

  // Each voxel's mean affinity over those channels (np.mean over axis 0:
  // the channels added in order, then divided), and from it each fragment's
  // mean (ndimage.mean: the voxels summed in C order, then divided). A
  // fragment whose mean is below the bias is background.
  const voxelMean = new Float64Array(N);
  for (const c of channels) {
    const plane = c * N;
    for (let i = 0; i < N; i++) voxelMean[i] += aff[plane + i]!;
  }
  for (let i = 0; i < N; i++) voxelMean[i] /= channels.length;
  const sum = new Float64Array(N + 1), count = new Uint32Array(N + 1);  // the ids are at most N
  for (let i = 0; i < N; i++) {
    const id = labels[i]!;
    if (id === 0) continue;
    sum[id] += voxelMean[i]!;
    count[id]++;
  }
  // The kept fragments numbered 1..n in order of first appearance (fastremap.renumber).
  const renumbered = new Uint32Array(N + 1);  // old id -> new id, 0 while unassigned
  let next = 1;
  const out = new Uint32Array(N);
  for (let i = 0; i < N; i++) {
    const id = labels[i]!;
    if (id === 0 || !(sum[id]! / count[id]! >= bias)) continue;  // background
    if (renumbered[id] === 0) renumbered[id] = next++;
    out[i] = renumbered[id]!;
  }
  return removeSmall(out, minSize);
}

// --- the postprocessor -------------------------------------------------------

export const affinityPostprocessor: PostprocessorDef = {
  // The Python class declares dtype uint64, num_channels 1 and is_segmentation True.
  declares: () => ({ dtype: "uint64", channels: () => 1, isSegmentation: true }),
  make: (step: Step) => {
    const bias = asNumber(step.bias, DEFAULT_BIAS, "AffinityPostprocessor.bias");
    const neighborhood = parseNeighborhood(step.neighborhood);
    return ({ data, shape }, context) => {
      const [, z, y, x] = shape;
      // Integer input is the 0-255 that DefaultPostprocessor produces (the
      // usual chain), scaled back to [0, 1]; float input is already an
      // affinity in [0, 1] (straight after a SigmoidPostprocessor, say) and
      // dividing it by 255 as well would leave every edge near zero.
      const integer = !(data instanceof Float32Array || data instanceof Float64Array);
      const aff = asFloat64(data);
      if (integer) for (let i = 0; i < aff.length; i++) aff[i] = aff[i]! / 255;
      const labels = mutexWatershed({ data: aff, shape }, neighborhood, bias);
      // Unique across chunks: the Morton increment on every object.
      const increment = mortonIncrement(context.chunkIndex, context.chunkNumVoxels);
      for (let i = 0; i < labels.length; i++) {
        if (labels[i] !== 0n) labels[i] += increment;
      }
      return { data: labels, shape: [1, z, y, x] };
    };
  },
};
