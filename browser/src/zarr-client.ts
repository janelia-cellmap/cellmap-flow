import * as zarr from "zarrita";

export interface ArraySummary {
  shape: number[];
  chunks: number[];
  dtype: string;
  ndim: number;
}

export type ZarrArray = zarr.Array<zarr.DataType, zarr.FetchStore>;

export function normalizeZarrUrl(input: string): string {
  let url = input.trim();
  // Strip Neuroglancer-style protocol prefixes (zarr://, n5://, precomputed://).
  url = url.replace(/^(zarr|zarr2|zarr3|n5|precomputed):\/\//, "");
  // Translate cloud-bucket schemes to HTTPS so the browser fetch can use them.
  if (url.startsWith("s3://")) {
    const rest = url.slice(5);
    const i = rest.indexOf("/");
    const bucket = i === -1 ? rest : rest.slice(0, i);
    const key = i === -1 ? "" : rest.slice(i);
    return `https://${bucket}.s3.amazonaws.com${key}`;
  }
  if (url.startsWith("gs://")) {
    const rest = url.slice(5);
    const i = rest.indexOf("/");
    const bucket = i === -1 ? rest : rest.slice(0, i);
    const key = i === -1 ? "" : rest.slice(i);
    return `https://storage.googleapis.com/${bucket}${key}`;
  }
  return url;
}

export async function openArray(
  url: string,
  preferredVoxelSize?: [number, number, number],
): Promise<ZarrArray> {
  const u = new URL(normalizeZarrUrl(url));
  const base = `${u.origin}${u.pathname.replace(/\/$/, "")}`;
  const store = new zarr.FetchStore(base);

  // Try opening as an array first.
  try {
    return (await zarr.open(store, { kind: "array" })) as ZarrArray;
  } catch (arrErr) {
    // Fall back: maybe it's an OME-Zarr group; follow multiscales and pick
    // the level whose voxel size best matches preferredVoxelSize (or the
    // first level if none is preferred).
    try {
      const zattrsResp = await fetch(`${base}/.zattrs`);
      if (!zattrsResp.ok) throw arrErr;
      const attrs = (await zattrsResp.json()) as {
        multiscales?: {
          datasets?: {
            path?: string;
            coordinateTransformations?: { type: string; scale?: number[] }[];
          }[];
        }[];
      };
      const datasets = attrs.multiscales?.[0]?.datasets ?? [];
      if (datasets.length === 0) throw arrErr;

      const chosen = pickBestMultiscale(datasets, preferredVoxelSize);
      const subUrl = `${base.replace(/\/$/, "")}/${chosen.path}`;
      const subStore = new zarr.FetchStore(subUrl);
      return (await zarr.open(subStore, { kind: "array" })) as ZarrArray;
    } catch {
      throw new Error(
        `not a zarr array at ${base}. ` +
          `If this is an OME-Zarr group, append a scale path like '/s0'. ` +
          `Underlying error: ${(arrErr as Error).message}`,
      );
    }
  }
}

interface MultiscaleDataset {
  path?: string;
  coordinateTransformations?: { type: string; scale?: number[] }[];
}

function pickBestMultiscale(
  datasets: MultiscaleDataset[],
  preferred?: [number, number, number],
): { path: string; scale: number[] | undefined } {
  const usable = datasets.filter((d) => d.path);
  if (!preferred) {
    const d = usable[0];
    return { path: d.path!, scale: d.coordinateTransformations?.find((t) => t.type === "scale")?.scale };
  }
  // For each dataset, derive its voxel-scale (last 3 elements). Pick the one
  // closest to `preferred` in log-space.
  let best = usable[0];
  let bestErr = Infinity;
  for (const d of usable) {
    const scale = d.coordinateTransformations?.find((t) => t.type === "scale")?.scale;
    if (!scale) continue;
    const xyz = scale.slice(-3);
    const err = preferred.reduce((acc, p, i) => acc + Math.abs(Math.log2(xyz[i] / p)), 0);
    if (err < bestErr) { bestErr = err; best = d; }
  }
  const bestScale = best.coordinateTransformations?.find((t) => t.type === "scale")?.scale;
  return { path: best.path!, scale: bestScale };
}

export function summarize(arr: ZarrArray): ArraySummary {
  return {
    shape: Array.from(arr.shape),
    chunks: Array.from(arr.chunks),
    dtype: arr.dtype,
    ndim: arr.shape.length,
  };
}

export type NumericArray = Uint8Array | Uint16Array | Int16Array | Float32Array;

export async function readSlice2D(
  arr: ZarrArray,
  zIdx: number,
  y: [number, number],
  x: [number, number],
): Promise<{ data: NumericArray; height: number; width: number }> {
  const ndim = arr.shape.length;
  if (ndim < 2) throw new Error(`array has ndim=${ndim}, expected >= 2`);

  const sel: (zarr.Slice | number)[] = [];
  if (ndim >= 3) {
    sel.push(Math.max(0, Math.min(zIdx, arr.shape[ndim - 3] - 1)));
  }
  sel.push(zarr.slice(y[0], y[1]));
  sel.push(zarr.slice(x[0], x[1]));

  while (sel.length < ndim) sel.unshift(0);

  const chunk = await zarr.get(arr, sel);
  const [height, width] = chunk.shape;
  return {
    data: chunk.data as NumericArray,
    height,
    width,
  };
}

export async function readSubvolume3D(
  arr: ZarrArray,
  z: [number, number],
  y: [number, number],
  x: [number, number],
): Promise<{ data: NumericArray; depth: number; height: number; width: number }> {
  const ndim = arr.shape.length;
  if (ndim < 3) throw new Error(`array has ndim=${ndim}, expected >= 3`);

  const sel: (zarr.Slice | number)[] = [
    zarr.slice(z[0], z[1]),
    zarr.slice(y[0], y[1]),
    zarr.slice(x[0], x[1]),
  ];
  while (sel.length < ndim) sel.unshift(0);

  const chunk = await zarr.get(arr, sel);
  const [depth, height, width] = chunk.shape;
  return {
    data: chunk.data as NumericArray,
    depth,
    height,
    width,
  };
}

export function toFloat32(data: NumericArray): Float32Array {
  if (data instanceof Float32Array) return data;
  return Float32Array.from(data);
}
