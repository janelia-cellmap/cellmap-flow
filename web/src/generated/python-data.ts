// The constants and catalogs the browser app shares with Python, as the
// Python that owns them has them: web/build/export_data.py writes
// python-data.json beside this file from cellmap_flow (`npm run export:data`,
// and `npm run render`, regenerate it; tests/web/test_python_data_fresh.py
// fails when it is stale). Never edit the JSON by hand: change the Python.
//
// PYTHON is the JSON with the type TypeScript infers from it, so a field the
// Python side renames or drops fails the typecheck where it is read. The
// two helpers turn the values that are Python syntax into their TypeScript
// equivalents: str.format templates (the shaders, the prompt's sentences)
// and compiled regular expressions (the finetune log's patterns).
import data from "./python-data.json";

export const PYTHON = data;

/** Python's `template.format(**values)` for the templates in the export:
 * each `{name}` becomes its value and `{{`, `}}` one brace. A name without a
 * value throws, as Python's KeyError does. Values are strings: a number is
 * formatted by the caller as the Python side formats it. */
export function pyFormat(template: string, values: Readonly<Record<string, string>> = {}): string {
  return template.replace(/\{\{|\}\}|\{(\w+)\}/g, (match: string, name: string | undefined) => {
    if (name === undefined) return match[0];
    if (!Object.hasOwn(values, name)) throw new Error(`no value for {${name}} in ${JSON.stringify(template)}`);
    return values[name];
  });
}

/** A Python pattern as the export gives it ({source, flags}), as a RegExp. */
export function pyRegExp(pattern: { source: string; flags: string }): RegExp {
  return new RegExp(pattern.source, pattern.flags);
}
