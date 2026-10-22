// FileSystemStore over a fake directory handle: paths into subdirectories,
// a missing file as undefined, and the directory handles it walked to kept.
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

describe("FileSystemStore keeps the directory handles it walked to", () => {
  it("looks a directory up once however many files of it are read, and retries one that was missing", async () => {
    const calls: string[] = [];
    const tree: Tree = { s0: { "0": { "0": { "0": new Uint8Array([1]), "1": new Uint8Array([2]) } } } };
    const counted = (d: DirectoryLike, path: string): DirectoryLike => ({
      async getDirectoryHandle(name) { calls.push(`${path}/${name}`); return counted(await d.getDirectoryHandle(name), `${path}/${name}`); },
      getFileHandle: (name) => d.getFileHandle(name),
    });
    const store = new FileSystemStore(counted(dir(tree), ""));
    expect(Array.from((await store.get("s0/0/0/0"))!)).toEqual([1]);
    expect(Array.from((await store.get("s0/0/0/1"))!)).toEqual([2]);
    expect(calls).toEqual(["/s0", "/s0/0", "/s0/0/0"]);  // the second read walked nothing
    expect(await store.get("s1/0/0/0")).toBeUndefined();  // s1 is not there (a missing chunk)
    tree.s1 = { "0": { "0": { "0": new Uint8Array([3]) } } };
    expect(Array.from((await store.get("s1/0/0/0"))!)).toEqual([3]);  // a directory made later is found
  });
});
