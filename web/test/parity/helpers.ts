// Shared by the parity tests: loading a fixture that
// test/fixtures/make_fixtures.py generated from the Python classes, and
// comparing arrays the way the fixtures ask (floats with a tolerance, since
// numpy computes in float32 where JavaScript may use float64; integers
// exactly).
import { readFileSync } from "node:fs";

import { expect } from "vitest";

/** An array as the fixtures carry it: numpy's dtype name, the shape, and the flat C-order values. */
export interface ArrayRecord {
  dtype: string;
  shape: number[];
  data: number[];
}

export type TypedArray =
  | Uint8Array
  | Uint16Array
  | Uint32Array
  | Int8Array
  | Int16Array
  | Int32Array
  | Float32Array
  | Float64Array
  | BigUint64Array
  | BigInt64Array;

export function loadFixture<T>(name: string): T {
  const url = new URL(`../fixtures/${name}`, import.meta.url);
  return JSON.parse(readFileSync(url, "utf8")) as T;
}

/** numpy's name for a typed array's element type. */
export function dtypeOf(array: ArrayLike<number> | ArrayLike<bigint>): string {
  const names: Record<string, string> = {
    Uint8Array: "uint8",
    Uint8ClampedArray: "uint8",
    Uint16Array: "uint16",
    Uint32Array: "uint32",
    Int8Array: "int8",
    Int16Array: "int16",
    Int32Array: "int32",
    Float32Array: "float32",
    Float64Array: "float64",
    BigUint64Array: "uint64",
    BigInt64Array: "int64",
  };
  const name = (array as object).constructor.name;
  return names[name] ?? name;
}

export const FLOAT_DTYPES = new Set(["float16", "float32", "float64"]);

/** Relative tolerance for float32 results: the Python computes in float32,
 * the TypeScript may compute in float64 and round once at the end, so the two
 * agree to a few float32 ulps (~1e-7 relative), far inside this. */
export const FLOAT_REL_TOL = 1e-5;
/** Absolute floor for values at or near zero, where a relative tolerance means nothing. */
export const FLOAT_ABS_TOL = 1e-6;

function describeMismatch(label: string, i: number, actual: number | bigint, expected: number): string {
  return `${label}: element ${i} is ${actual}, the Python gave ${expected}`;
}

/** Every element of `actual` equals the fixture's `expected`: exactly for
 * integer dtypes, within the float tolerance for float ones. */
export function expectArrayMatches(actual: ArrayLike<number> | ArrayLike<bigint>, expected: ArrayRecord, label: string): void {
  expect(actual.length, `${label}: length`).toBe(expected.data.length);
  const floats = FLOAT_DTYPES.has(expected.dtype);
  for (let i = 0; i < expected.data.length; i++) {
    const a = actual[i] as number | bigint;
    const e = expected.data[i]!;
    if (typeof a === "bigint") {
      if (a !== BigInt(e)) throw new Error(describeMismatch(label, i, a, e));
    } else if (floats) {
      const tol = FLOAT_REL_TOL * Math.max(Math.abs(a), Math.abs(e)) + FLOAT_ABS_TOL;
      if (!(Math.abs(a - e) <= tol)) throw new Error(describeMismatch(label, i, a, e));
    } else if (a !== e) {
      throw new Error(describeMismatch(label, i, a, e));
    }
  }
}

/** Whether `collection` (an array, a Set, or an object keyed by name, as a
 * module may choose to export its supported ops) names `name`. */
export function supports(collection: unknown, name: string): boolean {
  if (Array.isArray(collection)) return collection.includes(name);
  if (collection instanceof Set) return collection.has(name);
  if (collection instanceof Map) return collection.has(name);
  if (collection && typeof collection === "object") return name in collection;
  return false;
}
