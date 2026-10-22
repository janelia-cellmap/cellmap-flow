// What the browser keeps of cellmap_flow/ai_annotate/geometry.py: the names
// the viewer's side reads the plane by (viewer/state.ts depthAxisFromLayout,
// which reads the page's Neuroglancer as the server's depth_axis_from_layout
// reads its viewer) and Python's `:g` number format, which the prompt spells
// its resolution with (prompts.ts). Which plane is read and which voxels are
// written is the Python both modes run (cellmap_flow/portable/ai_annotate.py
// plan_plane, called from plane.ts).

export const AXIS_NAMES: readonly ["z", "y", "x"] = ["z", "y", "x"];

/** Neuroglancer names its cross-section panels after display-dimension
 * *slots*, not after the dimensions' names: the "xy" panel shows display
 * dimensions 0 and 1 and looks along 2, "xz" (rotated about the first)
 * looks along 1, "yz" (rotated about the second) along 0
 * (data_panel_layout.ts's AXES_RELATIVE_ORIENTATION). The display dimensions
 * are the first three of the viewer's coordinate space unless set.
 * cellmap-flow's viewers order their dimensions x, y, z (viewer/layers.ts
 * VIEWER_AXES), so the "xy" panel is the XY plane, looking along z. A viewer
 * in z, y, x order (a link made before) has a z-y slice looking along x
 * there, and the XY plane is its "yz" panel (geometry.py _LAYOUT_DEPTH_SLOT). */
export const LAYOUT_DEPTH_SLOT: Readonly<Record<string, number>> = { xy: 2, xz: 1, yz: 0 };

/** The data axis (0 z, 1 y, 2 x) display slot `slot` shows, or -1 when it shows none of them. */
export function slotAxis(names: readonly string[], slot: number): number {
  return slot < names.length ? (AXIS_NAMES as readonly string[]).indexOf(names[slot]) : -1;
}

/** Python's `format(value, "g")` with `precision` significant digits (6, as
 * `f"{x:g}"`): the shortest of fixed and scientific notation, trailing zeros
 * dropped, the exponent at least two digits; the prompt's resolution
 * sentence spells its number with it. A value exactly halfway at the last
 * digit may round the other way than Python (which rounds half to even
 * where JavaScript's toExponential rounds half up). */
export function formatG(value: number, precision = 6): string {
  if (Number.isNaN(value)) return "nan";
  if (!Number.isFinite(value)) return value > 0 ? "inf" : "-inf";
  if (value === 0) return Object.is(value, -0) ? "-0" : "0";
  const p = Math.max(precision, 1);
  const sign = value < 0 ? "-" : "";
  const magnitude = Math.abs(value);
  const scientific = magnitude.toExponential(p - 1);  // "d.ddddde+N", rounded to p digits
  const [mantissa, exponentText] = scientific.split("e");
  const exponent = Number(exponentText);
  if (exponent < -4 || exponent >= p) {
    const digits = stripZeros(mantissa);
    const expSign = exponent < 0 ? "-" : "+";
    return `${sign}${digits}e${expSign}${String(Math.abs(exponent)).padStart(2, "0")}`;
  }
  return sign + stripZeros(magnitude.toFixed(p - 1 - exponent));
}

/** A decimal string without its trailing zeros after the point, and without the point when nothing follows it. */
function stripZeros(text: string): string {
  return text.includes(".") ? text.replace(/0+$/, "").replace(/\.$/, "") : text;
}
