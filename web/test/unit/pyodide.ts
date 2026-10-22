// cellmap_flow.portable in node, for the tests of what calls it (the rank
// and Label it routes): Pyodide from the npm package (the version
// chunkmirage's workers load from jsDelivr), numpy from jsDelivr,
// build/portable.ts's modules written as the page's workers get them, and
// chunkmirage's own dispatch (chunkmirage.calls, from the clone) marshalling
// each answer as its worker does. scipy is loaded as the worker loads it:
// not at the start, but the first time a call fails to import it, and that
// call is made again. A wheel needs the network the first time (Pyodide then
// caches it in node_modules/pyodide): without it portablePython() is null,
// with a warning, and the tests that need it skip.
import { readFileSync } from "node:fs";
import { resolve } from "node:path";

import { PORTABLE_MODULES } from "../../build/portable";

/** Inference.python's shape (src/contracts.ts). */
export type PythonCall = <T>(fn: string, args: Record<string, unknown>, arrays?: ArrayBuffer[]) => Promise<T>;

const CALLS = resolve(import.meta.dirname, "../../chunkmirage/src/chunkmirage/calls.py");
/** The packages the worker loads on first import (pyworker.ts LAZY), by module name. */
const LAZY: Record<string, string> = { scipy: "scipy", skimage: "scikit-image" };

let loading: Promise<PythonCall | null> | null = null;

export function portablePython(): Promise<PythonCall | null> {
  return (loading ??= load().catch((e: unknown) => {
    console.warn(`[pyodide] not available in this run, so the tests that call Python skip: ${(e as Error).message}`);
    return null;
  }));
}

async function load(): Promise<PythonCall> {
  const { loadPyodide } = await import("pyodide");
  const py = await loadPyodide();
  await py.loadPackage(["numpy"]);
  const files = { ...PORTABLE_MODULES, "chunkmirage/__init__.py": "", "chunkmirage/calls.py": readFileSync(CALLS, "utf8") };
  for (const [path, text] of Object.entries(files)) {
    py.FS.mkdirTree(`/page/${path.slice(0, path.lastIndexOf("/"))}`);
    py.FS.writeFile(`/page/${path}`, text);
  }
  // As the worker's glue calls it: JavaScript's buffers to Python's first.
  py.runPython('import sys; sys.path.insert(0, "/page")\nfrom chunkmirage import calls\n\ndef call(fn, args, arrays):\n    return calls.call(fn, args, [a.to_py() for a in arrays])\n');
  const call = py.globals.get("call");
  const buffer = (b: Uint8Array) => b.buffer.slice(b.byteOffset, b.byteOffset + b.byteLength) as ArrayBuffer;
  // chunkmirage's pyworker.ts withPackages: a package the call could not
  // import (scipy, for Label it's components), loaded once, then the call again.
  const loading = new Map<string, Promise<unknown>>();
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const withPackages = async (run: () => any) => {
    for (let tries = 0; ; tries++) {
      try {
        return run();
      } catch (e) {
        const name = /No module named '(\w+)/.exec(String((e as Error)?.message ?? ""))?.[1];
        if (!name || !LAZY[name] || tries > 1) throw e;
        if (!loading.has(name)) loading.set(name, py.loadPackage([LAZY[name]]));
        await loading.get(name);
        py.runPython(`import sys\nfor m in [m for m in sys.modules if m == "${name}" or m.startswith("${name}.")]: del sys.modules[m]`);
      }
    }
  };
  return async <T>(fn: string, args: Record<string, unknown>, arrays: ArrayBuffer[] = []): Promise<T> => {
    const out = await withPackages(() => call(fn, JSON.stringify(args), arrays.map((b) => new Uint8Array(b))));
    if (typeof out === "string") return JSON.parse(out) as T;
    const js = out.toJs() as Uint8Array | [string, Uint8Array[]];
    out.destroy();
    if (js instanceof Uint8Array) return buffer(js) as T;
    return { value: JSON.parse(js[0]), arrays: js[1].map(buffer) } as T;
  };
}
