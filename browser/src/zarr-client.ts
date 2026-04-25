import * as zarr from "zarrita";

export interface ArraySummary {
  shape: number[];
  chunks: number[];
  dtype: string;
  ndim: number;
}

export type ZarrArray = zarr.Array<zarr.DataType, zarr.FetchStore>;

export async function openArray(url: string): Promise<ZarrArray> {
  const u = new URL(url);
  const base = `${u.origin}${u.pathname.replace(/\/$/, "")}`;
  const store = new zarr.FetchStore(base);
  return zarr.open(store, { kind: "array" }) as Promise<ZarrArray>;
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
