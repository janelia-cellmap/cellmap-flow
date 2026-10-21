// The Python both modes run (cellmap_flow/portable), bundled as text for
// chunkmirage's Pyodide workers, as chunkmirage bundles its own: Engine.start's
// `modules` writes these files into each worker beside chunkmirage's, and
// Inference.python calls them by module path (cellmap_flow.portable.calls).
// The package's own __init__.py is a stub here, as chunkmirage's is: the real
// one is the whole package's, and nothing under portable imports it.
/// <reference types="vite/client" />
const PY = import.meta.glob("../../cellmap_flow/portable/**/*.py", { query: "?raw", import: "default", eager: true }) as Record<string, string>;

/** Each module's path on the workers' Python path (`cellmap_flow/portable/rank.py`), to its text. */
export const PORTABLE_MODULES: Record<string, string> = {
  ...Object.fromEntries(Object.entries(PY).map(([path, text]) => [path.slice(path.indexOf("/cellmap_flow/") + 1), text])),
  "cellmap_flow/__init__.py": '"""cellmap-flow\'s portable Python, for the browser."""\n',
};
