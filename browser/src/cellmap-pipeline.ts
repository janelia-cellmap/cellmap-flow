// TypeScript ports of cellmap-flow's normalizer / postprocessor classes from
// `cellmap_flow/norm/input_normalize.py` and `cellmap_flow/post/postprocessors.py`.
// Class names + constructor parameters match the Python originals exactly so a
// pipeline JSON exported by cellmap-flow's pipeline builder can be loaded
// here unchanged.
//
// Phase 1 covers the common normalizers + postprocessors that don't depend
// on heavy C extensions (edt, mwatershed, fastremap, fastmorph). Anything
// else throws "not implemented in browser" with a pointer to install the
// full Python cellmap-flow.

// ---- input_normalize.py ----

export interface NormSpec {
  name: string;
  // Constructor kwargs from cellmap-flow's Python class.
  [key: string]: unknown;
}

export interface PostSpec {
  name: string;
  [key: string]: unknown;
}

export interface PostContext {
  chunkCorner: [number, number, number]; // chunk index in (z,y,x)
  chunkNumVoxels: number;                // voxels in a full chunk
}

abstract class InputNormalizer {
  abstract process(data: Float32Array): Float32Array;
}

export class MinMaxNormalizer extends InputNormalizer {
  min_value: number;
  max_value: number;
  invert: boolean;
  constructor(args: { min_value?: number; max_value?: number; invert?: boolean | string } = {}) {
    super();
    this.min_value = Number(args.min_value ?? 0);
    this.max_value = Number(args.max_value ?? 255);
    const inv = args.invert ?? false;
    this.invert = typeof inv === "string" ? inv.toLowerCase() === "true" : Boolean(inv);
  }
  process(data: Float32Array): Float32Array {
    const range = this.max_value - this.min_value || 1;
    const out = new Float32Array(data.length);
    for (let i = 0; i < data.length; i++) {
      let v = data[i];
      if (v < this.min_value) v = this.min_value;
      else if (v > this.max_value) v = this.max_value;
      v = (v - this.min_value) / range;
      out[i] = this.invert ? 1 - v : v;
    }
    return out;
  }
}

export class ZScoreNormalizer extends InputNormalizer {
  mean: number;
  std: number;
  constructor(args: { mean?: number; std?: number } = {}) {
    super();
    this.mean = Number(args.mean ?? 0);
    this.std = Number(args.std ?? 1);
  }
  process(data: Float32Array): Float32Array {
    const out = new Float32Array(data.length);
    const inv = 1 / (this.std || 1);
    for (let i = 0; i < data.length; i++) out[i] = (data[i] - this.mean) * inv;
    return out;
  }
}

const NORMALIZER_REGISTRY: Record<string, new (args: Record<string, unknown>) => InputNormalizer> = {
  MinMaxNormalizer: MinMaxNormalizer as never,
  ZScoreNormalizer: ZScoreNormalizer as never,
};

export function makeNormalizer(spec: NormSpec): InputNormalizer {
  const Cls = NORMALIZER_REGISTRY[spec.name];
  if (!Cls) {
    throw new Error(
      `normalizer '${spec.name}' not implemented in the browser version. ` +
        `Supported: ${Object.keys(NORMALIZER_REGISTRY).join(", ")}. ` +
        `(Python-only normalizers: Dilate, EuclideanDistance, LambdaNormalizer.)`,
    );
  }
  const args: Record<string, unknown> = {};
  for (const [k, v] of Object.entries(spec)) if (k !== "name") args[k] = v;
  return new Cls(args);
}

// ---- postprocessors.py ----

abstract class PostProcessor {
  /** Output dtype of THIS step. */
  abstract dtype(): "uint8" | "uint16" | "uint64" | "float32";
  /** Number of output channels (defaults to passing through). */
  numChannels(inputChannels: number): number {
    return inputChannels;
  }
  /** Process Float32 data shaped (C, Dz, Dy, Dx) flattened. May reduce C. */
  abstract process(data: Float32Array, channels: number, spatial: number, ctx: PostContext): {
    data: Float32Array;
    channels: number;
  };
}

export class DefaultPostprocessor extends PostProcessor {
  clip_min: number;
  clip_max: number;
  bias: number;
  multiplier: number;
  constructor(args: { clip_min?: number; clip_max?: number; bias?: number; multiplier?: number } = {}) {
    super();
    this.clip_min = Number(args.clip_min ?? -1.0);
    this.clip_max = Number(args.clip_max ?? 1.0);
    this.bias = Number(args.bias ?? 1.0);
    this.multiplier = Number(args.multiplier ?? 127.5);
  }
  dtype() { return "uint8" as const; }
  process(data: Float32Array, channels: number) {
    for (let i = 0; i < data.length; i++) {
      let v = data[i];
      if (v < this.clip_min) v = this.clip_min;
      else if (v > this.clip_max) v = this.clip_max;
      data[i] = (v + this.bias) * this.multiplier;
    }
    return { data, channels };
  }
}

export class ThresholdPostprocessor extends PostProcessor {
  threshold: number;
  constructor(args: { threshold?: number } = {}) {
    super();
    this.threshold = Number(args.threshold ?? 0.5);
  }
  dtype() { return "uint8" as const; }
  process(data: Float32Array, channels: number) {
    for (let i = 0; i < data.length; i++) data[i] = data[i] > this.threshold ? 1 : 0;
    return { data, channels };
  }
}

export class ChannelSelection extends PostProcessor {
  channels: number[];
  constructor(args: { channels?: string | number[] } = {}) {
    super();
    const c = args.channels ?? "0";
    if (typeof c === "string") this.channels = c.split(",").map((s) => parseInt(s.trim(), 10));
    else this.channels = c.map((n) => Number(n));
  }
  dtype() { return "uint8" as const; }
  numChannels() { return this.channels.length; }
  process(data: Float32Array, channels: number, spatial: number) {
    const out = new Float32Array(this.channels.length * spatial);
    for (let i = 0; i < this.channels.length; i++) {
      const src = this.channels[i] * spatial;
      out.set(data.subarray(src, src + spatial), i * spatial);
    }
    void channels;
    return { data: out, channels: this.channels.length };
  }
}

const POSTPROCESSOR_REGISTRY: Record<string, new (args: Record<string, unknown>) => PostProcessor> = {
  DefaultPostprocessor: DefaultPostprocessor as never,
  ThresholdPostprocessor: ThresholdPostprocessor as never,
  ChannelSelection: ChannelSelection as never,
};

export function makePostprocessor(spec: PostSpec): PostProcessor {
  const Cls = POSTPROCESSOR_REGISTRY[spec.name];
  if (!Cls) {
    throw new Error(
      `postprocessor '${spec.name}' not implemented in the browser version. ` +
        `Supported: ${Object.keys(POSTPROCESSOR_REGISTRY).join(", ")}. ` +
        `(Python-only: LabelPostprocessor, AffinityPostprocessor, SimpleBlockwiseMerger, ` +
        `MortonSegmentationRelabeling, LambdaPostprocessor.)`,
    );
  }
  const args: Record<string, unknown> = {};
  for (const [k, v] of Object.entries(spec)) if (k !== "name") args[k] = v;
  return new Cls(args);
}

export function listSupported(): { normalizers: string[]; postprocessors: string[] } {
  return {
    normalizers: Object.keys(NORMALIZER_REGISTRY),
    postprocessors: Object.keys(POSTPROCESSOR_REGISTRY),
  };
}
