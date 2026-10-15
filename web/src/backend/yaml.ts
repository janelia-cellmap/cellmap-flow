// A YAML emitter for the exported config (GET /api/export-config), writing
// what PyYAML's `yaml.dump(value, default_flow_style=False, sort_keys=False)`
// writes for the flat shapes the config has: mappings of scalars, lists of
// mappings, lists of scalars. That keeps a browser export byte-compatible
// with a server's, and readable by `cellmap_flow yaml`. No dependency: the
// shape does not need one.
//
// PyYAML's block style, as copied here: a list inside a mapping starts at the
// key's own indentation ("indentless" sequences), a mapping in a list item
// puts its first key after "- " and the rest two spaces in, an empty list is
// `[]` and an empty mapping `{}`. A string is written plain unless YAML would
// read it as something else (a number, a boolean, null, a date) or its
// characters need quoting; then it is double-quoted with JSON escapes, which
// YAML reads the same way PyYAML's single quotes do.

export type YamlValue = string | number | boolean | null | undefined | YamlValue[] | { [key: string]: YamlValue };

export function toYaml(value: YamlValue): string {
  if (!isMapping(value) && !Array.isArray(value)) return `${scalar(value)}\n`;
  if (isMapping(value) && Object.keys(value).length === 0) return "{}\n";
  if (Array.isArray(value) && value.length === 0) return "[]\n";
  return block(value, 0).join("\n") + "\n";
}

/** The lines of a mapping or list at `indent` spaces. */
function block(value: YamlValue, indent: number): string[] {
  const pad = " ".repeat(indent);
  const lines: string[] = [];
  if (Array.isArray(value)) {
    for (const item of value) {
      if (isMapping(item) && Object.keys(item).length) {
        // "- key: value" for the first key, the rest indented under it
        const inner = block(item, indent + 2);
        lines.push(`${pad}- ${inner[0].slice(indent + 2)}`, ...inner.slice(1));
      } else if (Array.isArray(item) && item.length) {
        const inner = block(item, indent + 2);
        lines.push(`${pad}- ${inner[0].slice(indent + 2)}`, ...inner.slice(1));
      } else lines.push(`${pad}- ${scalar(item)}`);
    }
    return lines;
  }
  for (const [key, item] of Object.entries(value as { [key: string]: YamlValue })) {
    const k = scalar(key);
    if (isMapping(item) && Object.keys(item).length) lines.push(`${pad}${k}:`, ...block(item, indent + 2));
    else if (Array.isArray(item) && item.length) lines.push(`${pad}${k}:`, ...block(item, indent)); // indentless
    else lines.push(`${pad}${k}: ${scalar(item)}`);
  }
  return lines;
}

function isMapping(value: YamlValue): value is { [key: string]: YamlValue } {
  return value !== null && typeof value === "object" && !Array.isArray(value);
}

/** A scalar as YAML text: empty containers included, since a key with an
 * empty list or mapping is written on one line. */
function scalar(value: YamlValue): string {
  if (value === null || value === undefined) return "null";
  if (typeof value === "boolean") return value ? "true" : "false";
  if (typeof value === "number") return Number.isFinite(value) ? String(value) : value > 0 ? ".inf" : value < 0 ? "-.inf" : ".nan";
  if (Array.isArray(value)) return "[]";
  if (typeof value === "object") return "{}";
  return plainSafe(value) ? value : JSON.stringify(value);
}

/** Whether YAML reads `s` back as exactly this string when written plain. */
const RESOLVES_TO_OTHER_TYPE = [
  /^(true|false|yes|no|on|off|null|~)$/i, // YAML 1.1 booleans and null, as PyYAML resolves them (not y/n: PyYAML does not)
  /^[-+]?(\d[\d_]*)?(\.\d*)?([eE][-+]?\d+)?$/, // ints and floats (10, 1.5, .5, 1e3, 1_000)
  /^[-+]?0x[0-9a-fA-F_]+$|^[-+]?0o?[0-7_]+$|^[-+]?0b[01_]+$/, // other integer bases
  /^[-+]?\.(inf|Inf|INF)$|^\.(nan|NaN|NAN)$/,
  /^[-+]?\d+(:[0-5]?\d)+(\.\d*)?$/, // sexagesimal (1:30)
  /^\d{4}-\d\d?-\d\d?/, // dates and timestamps
];

export function plainSafe(s: string): boolean {
  if (s === "" || s !== s.trim()) return false;
  if (/[\n\r\t\u0000-\u001f\u007f]/.test(s)) return false;
  if (/^[-?:,[\]{}#&*!|>'"%@`]/.test(s)) return false; // indicators, which start other things
  if (/: | #/.test(s) || s.endsWith(":")) return false; // a key, or a comment
  if (s === "-" || s.startsWith("- ")) return false;
  return !RESOLVES_TO_OTHER_TYPE.some((re) => re.test(s));
}
