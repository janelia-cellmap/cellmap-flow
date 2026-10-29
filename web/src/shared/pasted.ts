// A dataset path as a person pastes it, as the app reads it: the browser's
// twin of cellmap_flow/portable/pasted.py, through the same pattern
// (exported by build/export_data.py), for the dataset field
// (backend/routes/data.ts) and a link's ?dataset= (link.ts parseLink). A path
// copied from Neuroglancer names its format too, in front (zarr://, zarr2://,
// zarr3://, n5://) or after it (|zarr2:, |zarr3:, |n5:, ...); the format is
// dropped, a path after its colon joined to the location (x.zarr|zarr2:raw is
// x.zarr/raw), and a precomputed volume named precomputed://<location>, as
// the app names one. Both twins are held to web/test/fixtures/pasted_paths.json.
import { PYTHON, pyRegExp } from "../generated/python-data";

/** The text, split into the location, the format named after it and the path after that format's colon. */
const NEUROGLANCER_SOURCE = pyRegExp(PYTHON.pasted.neuroglancer_source);

/** The dataset `text` names, without the Neuroglancer format around it. */
export function pastedPath(text: string): string {
  const [, location, formatAfter, inner] = NEUROGLANCER_SOURCE.exec(text)!;
  const path = inner ? `${location.replace(/\/+$/, "")}/${inner}` : location;
  return formatAfter === "neuroglancer-precomputed" ? `precomputed://${path}` : path;
}
