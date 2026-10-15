// FileSystemStore over a fake directory handle: paths into subdirectories,
// a missing file as undefined, other errors through.
import { describe, expect, it } from "vitest";

import { FileSystemStore, type DirectoryLike } from "../../src/state/local-store";

type Tree = { [name: string]: Tree | Uint8Array };

function domError(name: string) {
  const e = new Error(name);
  e.name = name;
  return e;
}

/** A directory handle over a nested object: a Uint8Array is a file. */
function dir(tree: Tree): DirectoryLike {
  return {
    async getDirectoryHandle(name) {
      const child = tree[name];
      if (child === undefined) throw domError("NotFoundError");
      if (child instanceof Uint8Array) throw domError("TypeMismatchError");
      return dir(child);
    },
    async getFileHandle(name) {
      const child = tree[name];
      if (child === undefined) throw domError("NotFoundError");
      if (!(child instanceof Uint8Array)) throw domError("TypeMismatchError");
      return { getFile: async () => ({ arrayBuffer: async () => child.buffer.slice(child.byteOffset, child.byteOffset + child.byteLength) as ArrayBuffer }) };
    },
  };
}

describe("FileSystemStore", () => {
  const store = new FileSystemStore(
    dir({
      "zarr.json": new TextEncoder().encode('{"zarr_format":3}'),
      s0: { c: { "0": { "0": { "0": new Uint8Array([7, 8, 9]) } } } },
    }),
  );

  it("reads a file by its path, through subdirectories, with or without a leading slash", async () => {
    expect(new TextDecoder().decode(await store.get("zarr.json"))).toBe('{"zarr_format":3}');
    expect(Array.from((await store.get("/s0/c/0/0/0"))!)).toEqual([7, 8, 9]);
  });

  it("gives undefined for a missing file (a chunk at the fill value), a wrong kind, or an empty path", async () => {
    expect(await store.get("s0/c/0/0/1")).toBeUndefined();
    expect(await store.get("s0/zarr.json")).toBeUndefined();
    expect(await store.get("zarr.json/x")).toBeUndefined(); // a file where a directory was expected
    expect(await store.get("")).toBeUndefined();
  });

  it("lets other errors through", async () => {
    const broken = new FileSystemStore({
      getDirectoryHandle: async () => {
        throw domError("NotAllowedError");
      },
      getFileHandle: async () => {
        throw domError("NotAllowedError");
      },
    });
    await expect(broken.get("zarr.json")).rejects.toThrow("NotAllowedError");
  });
});
