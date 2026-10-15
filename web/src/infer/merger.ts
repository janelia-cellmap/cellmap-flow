// Ids joined across chunk faces, reported as equivalences: the browser side
// of cellmap_flow/post/postprocessors.py's SimpleBlockwiseMerger.
//
// Every chunk is segmented on its own, so an object that crosses a chunk
// boundary has one id on each side. The step keeps, for every chunk it has
// seen, the ids on its six faces (the first and last plane along each axis,
// keyed by the offset to the neighbouring chunk the face touches); whenever
// the opposite face of that neighbour has been seen too, the ids that meet
// at the same face position are joined in an equivalence map, and the two
// faces are done with (the Python's keys_to_skip, dropped from its dict).
// The joined ids are reported through the layer's hook for the viewer's
// segmentation layer to show as equivalences, at most once every 5 s as
// the server posts them to the dashboard, but with the last state always
// following (the server, which sends on the next chunk only, has no trailing
// send). The block itself passes through, cast to uint64.
//
// The equivalence map is a port of neuroglancer.equivalence_map.EquivalenceMap
// as far as to_json() goes: the ids known to it grouped into their sets,
// each sorted, the sets sorted. The ids are uint64, so BigInt throughout;
// the hook gets them as decimal strings.
import type { Step, StepValue } from "../contracts";
import { asInt } from "./label";
import { toUint64 } from "./morton";
import type { Declares, LayerHooks, NdArray, NdData, PostContext, Postprocessor, PostprocessorDef } from "./post";

/** How often at most the merged ids are reported (chunkmirage_ops.EQUIVALENCES_EVERY_SECONDS). */
export const REPORT_EVERY_MS = 5000;

// ---------------------------------------------------------------------------
// The equivalence map
// ---------------------------------------------------------------------------

function compareIds(a: bigint, b: bigint): number {
  return a < b ? -1 : a > b ? 1 : 0;
}

/** Lexicographic order of two sorted id lists: element by element, a
 * shorter prefix first (Python's sorted() of lists). */
function compareSets(a: bigint[], b: bigint[]): number {
  const n = Math.min(a.length, b.length);
  for (let i = 0; i < n; i++) {
    const c = compareIds(a[i], b[i]);
    if (c) return c;
  }
  return a.length - b.length;
}

/** neuroglancer.equivalence_map.EquivalenceMap, as far as the merger uses
 * it: union-find over uint64 ids, every id ever passed to union() known to
 * it, and to_json(). */
export class EquivalenceMap {
  private parent = new Map<bigint, bigint>();
  private size = new Map<bigint, number>();

  /** How many ids the map knows. */
  get count(): number { return this.parent.size; }

  /** `a`'s representative; an unknown id becomes known, in a set of its own
   * (as _get_representative does). */
  find(a: bigint): bigint {
    let root = this.parent.get(a);
    if (root === undefined) {
      this.parent.set(a, a);
      this.size.set(a, 1);
      return a;
    }
    const path: bigint[] = [];
    let node = a;
    while (root !== node) {
      path.push(node);
      node = root;
      root = this.parent.get(node)!;
    }
    for (const n of path) this.parent.set(n, root);  // path compression
    return root;
  }

  /** Join the sets of `a` and `b` (both become known); true when they
   * were apart, so the partition changed. */
  union(a: bigint, b: bigint): boolean {
    let ra = this.find(a), rb = this.find(b);
    if (ra === rb) return false;
    if (this.size.get(ra)! < this.size.get(rb)!) [ra, rb] = [rb, ra];
    this.parent.set(rb, ra);
    this.size.set(ra, this.size.get(ra)! + this.size.get(rb)!);
    return true;
  }

  /** The known ids grouped into their sets, each sorted ascending, the
   * sets in lexicographic order (EquivalenceMap.sets() as to_json() sorts
   * them). An id only ever joined with itself is a set of one. */
  sets(): bigint[][] {
    const groups = new Map<bigint, bigint[]>();
    for (const id of this.parent.keys()) {
      const root = this.find(id);
      const group = groups.get(root);
      if (group) group.push(id);
      else groups.set(root, [id]);
    }
    const sets = [...groups.values()];
    for (const set of sets) set.sort(compareIds);
    return sets.sort(compareSets);
  }

  /** EquivalenceMap.to_json(): sets() with the ids as decimal strings. */
  toJSON(): string[][] {
    return this.sets().map((set) => set.map(String));
  }
}

// ---------------------------------------------------------------------------
// Faces
// ---------------------------------------------------------------------------

/** The six faces, by the offset [dz, dy, dx] to the neighbouring chunk each
 * touches, in the order of the Python's `self.slices`: -1 is the first
 * plane along that axis, 1 the last. */
export const FACE_OFFSETS: readonly (readonly [number, number, number])[] = [
  [-1, 0, 0], [1, 0, 0], [0, -1, 0], [0, 1, 0], [0, 0, -1], [0, 0, 1],
];

/** One id as a uint64 value, from whatever array the labels are in. */
function idOf(v: number | bigint): bigint {
  return typeof v === "bigint" ? v : BigInt(Math.trunc(v));
}

/** The plane of `labels` (shape [nz, ny, nx]) that the face at `offset`
 * is: `segmentation[slice]` in the Python, a 2-D array whose two
 * coordinates are the block's remaining axes in order (y, x for a z face;
 * z, x for a y face; z, y for an x face), in raster order. */
export function faceOf(labels: NdData, shape: readonly number[], offset: readonly [number, number, number]): { values: BigUint64Array; height: number; width: number } {
  const [nz, ny, nx] = shape;
  const axis = offset.findIndex((d) => d !== 0);
  const plane = offset[axis] < 0 ? 0 : shape[axis] - 1;
  let height: number, width: number;
  let at: (c0: number, c1: number) => number;
  if (axis === 0) { height = ny; width = nx; at = (y, x) => (plane * ny + y) * nx + x; }
  else if (axis === 1) { height = nz; width = nx; at = (z, x) => (z * ny + plane) * nx + x; }
  else { height = nz; width = ny; at = (z, y) => (z * ny + y) * nx + plane; }
  const values = new BigUint64Array(height * width);
  for (let c0 = 0; c0 < height; c0++) {
    for (let c1 = 0; c1 < width; c1++) values[c0 * width + c1] = idOf(labels[at(c0, c1)]);
  }
  return { values, height, width };
}

/** fastmorph.erode on a 2-D array of labels, `iterations` times: multilabel
 * erosion with a full 3×3 stencil, the border counting as background. A
 * voxel keeps its id only if all eight of its neighbours exist and carry
 * the same id; one at the plane's edge, or beside another id or the
 * background, becomes 0. (fastmorph 1.8's erode, checked against a
 * per-label scipy binary_erosion with border_value 0 on random planes.) */
export function erodeLabels2d(values: BigUint64Array, height: number, width: number, iterations: number): BigUint64Array {
  let current = values;
  for (let it = 0; it < iterations; it++) {
    const out = new BigUint64Array(current.length);
    for (let i = 1; i < height - 1; i++) {
      for (let j = 1; j < width - 1; j++) {
        const v = current[i * width + j];
        if (v === 0n) continue;
        let keep = true;
        for (let di = -1; di <= 1 && keep; di++) {
          for (let dj = -1; dj <= 1; dj++) {
            if (current[(i + di) * width + (j + dj)] !== v) { keep = false; break; }
          }
        }
        if (keep) out[i * width + j] = v;
      }
    }
    current = out;
  }
  return current;
}

/** The nonzero ids on a face, by face position "c0,c1" (the Python's
 * `(coord_0, coord_1)` from np.where), after `erosion` erosions. */
export function faceIds(labels: NdData, shape: readonly number[], offset: readonly [number, number, number], erosion: number): Map<string, bigint> {
  const face = faceOf(labels, shape, offset);
  const values = erosion > 0 ? erodeLabels2d(face.values, face.height, face.width, erosion) : face.values;
  const ids = new Map<string, bigint>();
  for (let c0 = 0; c0 < face.height; c0++) {
    for (let c1 = 0; c1 < face.width; c1++) {
      const v = values[c0 * face.width + c1];
      if (v !== 0n) ids.set(`${c0},${c1}`, v);
    }
  }
  return ids;
}

// ---------------------------------------------------------------------------
// The merger's state
// ---------------------------------------------------------------------------

/** A recorded face: which chunk, which side, and its ids. */
interface Face {
  chunk: readonly number[];
  offset: readonly [number, number, number];
  ids: Map<string, bigint>;
}

/** The Python's dict key `(chunk_corner, slice_reference)`. */
function faceKey(chunk: readonly number[], offset: readonly number[]): string {
  return `${chunk.join(",")}|${offset.join(",")}`;
}

/** The key of the face this one touches: the neighbour at `offset`, seen
 * from the other side. */
function neighbourKey(face: Face): string {
  return faceKey(face.chunk.map((c, a) => c + face.offset[a]), face.offset.map((d) => -d));
}

/** SimpleBlockwiseMerger's bookkeeping, apart from the step's cast and the
 * reporting: one per served layer, fed every chunk in whatever order the
 * viewer asks for them. */
export class BlockwiseMerger {
  readonly equivalences = new EquivalenceMap();
  /** chunk_slice_position_to_coords_id_dict: the faces not yet matched. */
  private faces = new Map<string, Face>();
  /** keys_to_skip: faces matched with their neighbour, never looked at again. */
  private done = new Set<string>();

  constructor(readonly channel: number, readonly erosion: number) {}

  /** How many faces are recorded and waiting for their neighbour. */
  get waiting(): number { return this.faces.size; }

  /** Record `nd`'s faces as chunk `chunkIndex`'s and join what meets a
   * recorded neighbour; true when the equivalences changed. */
  addChunk(nd: NdArray, chunkIndex: readonly number[]): boolean {
    if (nd.shape.length !== 4) throw new Error(`SimpleBlockwiseMerger expects a (c, z, y, x) block, got shape ${nd.shape.join("×")}`);
    const [c, ...spatial] = nd.shape;
    if (this.channel < 0 || this.channel >= c) throw new Error(`SimpleBlockwiseMerger: channel ${this.channel} of an output with ${c} channel(s)`);
    if (chunkIndex.length !== 3) throw new Error(`a chunk index has three axes (z, y, x), got ${JSON.stringify(chunkIndex)}`);
    const plane = spatial[0] * spatial[1] * spatial[2];
    const labels = nd.data.subarray(this.channel * plane, (this.channel + 1) * plane) as NdData;
    const chunk = [...chunkIndex];
    for (const offset of FACE_OFFSETS) {
      this.faces.set(faceKey(chunk, offset), { chunk, offset, ids: faceIds(labels, spatial, offset, this.erosion) });
    }
    // A chunk computed again (the cache dropped it) records its faces again;
    // those already matched go straight back out, as the Python pops its
    // keys_to_skip after every update.
    for (const key of this.done) this.faces.delete(key);
    return this.calculateEquivalences();
  }

  /** calculate_equivalences: for every waiting face whose neighbour's
   * opposite face is waiting too, join the ids at the same positions and
   * mark both done. */
  private calculateEquivalences(): boolean {
    let changed = false;
    for (const [key, face] of [...this.faces]) {
      if (this.done.has(key)) continue;
      const otherKey = neighbourKey(face);
      const other = this.faces.get(otherKey);
      // The Python tests the neighbour's dict for truth: a recorded face
      // with no nonzero voxel is empty, and so not matched (nor marked
      // done) until a face with ids looks across at it. A face is matched
      // from either side, so the result is the same whichever is empty;
      // this keeps the bookkeeping identical.
      if (!other || other.ids.size === 0) continue;
      if (this.done.has(otherKey)) continue;
      for (const [position, id1] of face.ids) {
        const id2 = other.ids.get(position);
        if (id2 !== undefined && this.equivalences.union(id1, id2)) changed = true;
      }
      this.done.add(key);
      this.done.add(otherKey);
    }
    return changed;
  }
}

// ---------------------------------------------------------------------------
// Reporting, throttled
// ---------------------------------------------------------------------------

export interface Throttled {
  /** Ask for a call: now if the last was `intervalMs` ago or more, else
   * once when that time has passed (a pending call is not doubled). */
  request(): void;
  /** Whether a call is scheduled. */
  pending(): boolean;
}

/** `send` at most once every `intervalMs`, the first time at once, and
 * always once more after the last request (a trailing call), so the state
 * `send` reads when it runs is the latest. Date.now() and setTimeout are
 * looked up when used, so fake timers in a test apply. */
export function throttled(send: () => void, intervalMs: number): Throttled {
  let lastSentAt = -Infinity;
  let timer: ReturnType<typeof setTimeout> | null = null;
  const fire = () => {
    timer = null;
    lastSentAt = Date.now();
    send();
  };
  return {
    request() {
      if (timer !== null) return;
      const wait = lastSentAt + intervalMs - Date.now();
      if (wait <= 0) fire();
      else timer = setTimeout(fire, wait);
    },
    pending: () => timer !== null,
  };
}

// ---------------------------------------------------------------------------
// The step
// ---------------------------------------------------------------------------

/** The constructor's `int(...)` of a parameter. */
function parameter(value: StepValue | undefined, fallback: number, what: string): number {
  const n = asInt(value, fallback, what);
  if (n < 0) throw new Error(`${what} must not be negative, got ${JSON.stringify(value)}`);
  return n;
}

/** The built step with its merger, for tests that read the state. */
export function makeMerger(step: Step, hooks: LayerHooks): { merger: BlockwiseMerger; process: Postprocessor } {
  const merger = new BlockwiseMerger(
    parameter(step.channel, 0, "SimpleBlockwiseMerger.channel"),
    parameter(step.face_erosion_iterations, 0, "SimpleBlockwiseMerger.face_erosion_iterations"),
  );
  const onEquivalences = hooks.onEquivalences;
  const report = onEquivalences
    ? throttled(() => {
      // As the server posts from a thread of its own: a listener that
      // fails must not fail the chunk.
      try {
        onEquivalences(merger.equivalences.toJSON());
      } catch (e) {
        console.warn("SimpleBlockwiseMerger: the equivalences listener failed", e);
      }
    }, REPORT_EVERY_MS)
    : null;
  const process: Postprocessor = (nd: NdArray, context: PostContext): NdArray => {
    if (merger.addChunk(nd, context.chunkIndex) && report) report.request();
    return { data: toUint64(nd.data), shape: nd.shape };  // data.astype(np.uint64)
  };
  return { merger, process };
}

/** SimpleBlockwiseMerger(channel=0, face_erosion_iterations=0). */
export const blockwiseMerger: PostprocessorDef = {
  declares: (): Declares => ({ dtype: "uint64", isSegmentation: true }),
  make: (step, hooks) => makeMerger(step, hooks).process,
};
