// Self-hosted Neuroglancer, bundled by Vite from the npm package. Mounts
// inline into #ng-host. Imports mirror tourguide's bundled_viewer.ts —
// Vite honors exports conditions but NOT self-imports, so we register
// every data source / kvstore / layer we care about explicitly here
// before calling setupDefaultViewer.

import "neuroglancer/unstable/kvstore/http/register_frontend.js";
import "neuroglancer/unstable/kvstore/s3/register_frontend.js";
import "neuroglancer/unstable/kvstore/gcs/register.js";
import "neuroglancer/unstable/kvstore/gzip/register.js";
import "neuroglancer/unstable/kvstore/byte_range/register.js";
import "neuroglancer/unstable/kvstore/zip/register_frontend.js";
import "neuroglancer/unstable/kvstore/ocdbt/register_frontend.js";
import "neuroglancer/unstable/kvstore/icechunk/register_frontend.js";
import "neuroglancer/unstable/kvstore/middleauth/register_frontend.js";
import "neuroglancer/unstable/kvstore/middleauth/register_credentials_provider.js";
import "neuroglancer/unstable/kvstore/ngauth/register.js";
import "neuroglancer/unstable/kvstore/ngauth/register_credentials_provider.js";

import "neuroglancer/unstable/datasource/zarr/register_default.js";
import "neuroglancer/unstable/datasource/n5/register_default.js";
import "neuroglancer/unstable/datasource/precomputed/register_default.js";
import "neuroglancer/unstable/datasource/render/register_default.js";
import "neuroglancer/unstable/datasource/nifti/register_default.js";
import "neuroglancer/unstable/datasource/obj/register_default.js";
import "neuroglancer/unstable/datasource/vtk/register_default.js";
import "neuroglancer/unstable/datasource/deepzoom/register_default.js";
import "neuroglancer/unstable/datasource/dvid/register_default.js";
import "neuroglancer/unstable/datasource/dvid/register_credentials_provider.js";

import "neuroglancer/unstable/layer/image/index.js";
import "neuroglancer/unstable/layer/segmentation/index.js";
import "neuroglancer/unstable/layer/annotation/index.js";
import "neuroglancer/unstable/layer/single_mesh/index.js";

import { setupDefaultViewer } from "neuroglancer/unstable/ui/default_viewer_setup.js";
import type { Viewer as NgViewer } from "neuroglancer/unstable/viewer.js";

let viewer: NgViewer | null = null;

export function mountNg(state: Record<string, unknown>): NgViewer {
  const target = document.getElementById("ng-host");
  if (!target) throw new Error("missing #ng-host element");

  if (!viewer) {
    viewer = setupDefaultViewer({ target }) as unknown as NgViewer;
  }
  viewer.state.restoreState(state);
  return viewer;
}
