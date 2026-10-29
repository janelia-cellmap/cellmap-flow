// The raw volume's channels: how many it has, what its metadata calls them,
// and how chunkmirage's reader reads one of them, so that a model, a
// training read, an AI annotation's plane and the ranking each read the
// channel the chain picks (norms.ts selectedChannel), as the server reads it
// (io/raw.py channel_select).
//
// The reader has two ways with channels (web/chunkmirage/web/src/sources.ts).
// A precomputed volume of several, or a stack://, is read whole: every block
// holds all its channels, one after another, and its SourceInfo counts them.
// A zarr image's leading axes (OME-Zarr's c, of the czyx light microscopy
// writes) are pinned when the source is opened, at index 0 unless a view's
// `select` names another, so its SourceInfo says one channel. For those the
// count and the axis's name are read here from the image's metadata, by
// chunkmirage's own openImage (so the name is the one `select` takes), and
// the labels from the group's omero metadata. The axis is the one the
// server's rule picks: the leading axis named as a channel axis is
// (PYTHON.channels.axis_names), else the first.
import type { SourceInfo } from "../../chunkmirage/web/src/types";
import { PYTHON } from "../generated/python-data";

export interface RawChannels {
  /** How many: 1 for a volume of one. */
  count: number;
  /** What the metadata calls each (OME-Zarr's omero.channels[].label),
   * when it names every one; else null. */
  labels: string[] | null;
  /** The zarr axis a channel is read on, by name (a view's `select` pins
   * it); null where the reader reads the channels together, or there is one. */
  axis: string | null;
}

const ONE: RawChannels = { count: 1, labels: null, axis: null };

/** The channels of the volume the engine opened at `url` as `source`. A
 * volume that is not an OME-Zarr image (a zarr array, an N5, a precomputed
 * one of one channel) is read as the reader opened it: one channel. */
export async function rawChannels(url: string, source: SourceInfo): Promise<RawChannels> {
  if (source.channels > 1) return { count: source.channels, labels: null, axis: null };
  // zarrita on the page only for this, the reader worker has its own
  const { fetchStore, openImage, zarr } = await import("../../chunkmirage/web/src/ome");
  let image: Awaited<ReturnType<typeof openImage>>;
  try {
    image = await openImage(url);
  } catch {
    return ONE;
  }
  const root = zarr.root(fetchStore(url.replace(/\/+$/, "")));
  const attrs = await zarr.open(root, { kind: "group" }).then((group) => (group.attrs ?? {}) as Record<string, unknown>, () => ({}));
  return zarrChannels(image.names, image.levels[0].fullShape, attrs);
}

/** The channels of a zarr image whose axes are `names` (z, y, x last), of
 * `shape` at its finest level (every axis), with group attributes `attrs`. */
export function zarrChannels(names: readonly string[], shape: readonly number[], attrs: Record<string, unknown>): RawChannels {
  const lead = names.slice(0, -3);
  const axis = lead.find((name) => PYTHON.channels.axis_names.includes(name)) ?? lead[0];
  const count = axis === undefined ? 1 : shape[names.indexOf(axis)];
  return { count, labels: omeroLabels(attrs, count), axis: count > 1 ? (axis ?? null) : null };
}

/** omero.channels[].label (OME-Zarr 0.5 keeps omero under `ome`), when it
 * names each of the `count` channels; else null (io/raw.py raw_channels). */
function omeroLabels(attrs: Record<string, unknown>, count: number): string[] | null {
  const omero = ((attrs.ome as Record<string, unknown> | undefined)?.omero ?? attrs.omero) as { channels?: unknown } | undefined;
  const channels = omero?.channels;
  if (!Array.isArray(channels) || channels.length !== count) return null;
  const labels = channels.map((channel) => (channel as { label?: unknown } | null)?.label);
  return labels.every((label) => typeof label === "string" && label) ? (labels as string[]) : null;
}
