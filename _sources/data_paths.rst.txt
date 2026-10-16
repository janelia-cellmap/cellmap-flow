Data paths
==========

Wherever cellmap-flow takes a dataset -- ``data_path`` in a YAML file,
``-d`` on the command line, the dashboard, a viewer layer's ``path`` -- it
takes the same kinds of path, and reads the same metadata and voxels from a
URL as from the same files on disk.

What a path can be
------------------

.. list-table::
   :header-rows: 1
   :widths: 22 78

   * - Format
     - Examples
   * - zarr v2, OME-NGFF or funlib attributes
     - ``/nrs/cellmap/data/jrc_hela-2/jrc_hela-2.zarr/recon-1/em/fibsem-uint8``

       ``s3://janelia-cosem-datasets/jrc_hela-2/jrc_hela-2.zarr/recon-1/em/fibsem-uint8/s1``

       ``https://janelia-cosem-datasets.s3.amazonaws.com/jrc_hela-2/jrc_hela-2.zarr/recon-1/em/fibsem-uint8``

       ``gs://bucket/data.zarr/raw``
   * - zarr v3 (OME-NGFF 0.5)
     - ``/data/v3.zarr/em``, or the same at an ``http(s)://``, ``s3://`` or ``gs://`` URL
   * - N5
     - ``/data/x.n5/em/fibsem-uint16``, ``s3://janelia-cosem-datasets/jrc_hela-2/jrc_hela-2.n5/em/fibsem-uint16/s0``
   * - neuroglancer precomputed
     - ``precomputed:///data/volume``, ``precomputed://gs://flyem-male-cns/em/em-clahe-jpeg``,
       ``precomputed://https://storage.googleapis.com/flyem-male-cns/em/em-clahe-jpeg``,
       ``precomputed://s3://bucket/volume``, and a bare ``gs://flyem-male-cns/em/em-clahe-jpeg``

Paths run on past their container into the group or array inside it. The
container ends at the last path component ending in ``.zarr`` or ``.n5``;
without one, it is the nearest directory up from the path with a ``.zgroup``
(on disk and at a URL alike), or the node's own ``zarr.json`` for zarr v3.

- A **multiscale group** (OME-NGFF ``multiscales``, or a precomputed volume's
  scales) is read at the level for the voxel size asked for, or at its first
  level when none is. A path to one level (``…/s1``, a precomputed volume's
  ``…/s2``) is read at that level.
- A **group without multiscales** is read at its first array: the first
  subdirectory, sorted, that is one on disk; at a URL, which cannot be listed,
  its ``s0``.
- A shell-escaped space (``my\ data.zarr``) is unescaped in a local path, never
  in a URL.

**A gs:// path is a precomputed volume unless it has a .zarr or .n5
component.** That is how cellmap-flow has always read ``gs://`` paths. A zarr
whose names lack the suffix (such as CMIP6's ``gs://cmip6/CMIP6/…/tas``) is
read through its public URL instead:
``https://storage.googleapis.com/cmip6/CMIP6/…/tas``.

Remote data and credentials
---------------------------

Everything is read with tensorstore: no extra packages are needed for any
scheme. Public data needs no credentials.

``s3://``
   Read anonymously first, so a public bucket is read whatever AWS
   credentials are set up: S3 refuses even a public read signed with an
   expired or foreign key. When the bucket refuses an anonymous read,
   cellmap-flow tries again with the AWS default credentials: the
   ``AWS_ACCESS_KEY_ID``/``AWS_SECRET_ACCESS_KEY`` (and ``AWS_SESSION_TOKEN``)
   environment variables, ``~/.aws/credentials`` and ``~/.aws/config``
   (``AWS_PROFILE`` picks the profile), then an EC2 instance's role.

   An S3-compatible store (MinIO, Ceph, Wasabi) is named with
   ``AWS_ENDPOINT_URL_S3`` or ``AWS_ENDPOINT_URL``, and its region with
   ``AWS_REGION`` or ``AWS_DEFAULT_REGION``.

``gs://``
   Read with Google application-default credentials when there are any:
   ``GOOGLE_APPLICATION_CREDENTIALS``, or the file
   ``gcloud auth application-default login`` writes. Without them the read is
   anonymous. When Google Cloud Storage refuses those credentials (an expired
   or revoked login refuses even public buckets), the bucket is read at its
   public URL, ``https://storage.googleapis.com/<bucket>/<path>``.

``http://`` and ``https://``
   Read as they are, with no credentials.

The way in that worked is remembered for each bucket, so only the first read
of a private bucket is refused. A failed metadata request is retried for
about 5 seconds before the path is reported unreadable.

The raw layer in the viewer
---------------------------

By default (``wrap_raw: true``) the raw layer is served by cellmap-flow
itself, through the same reader as inference, so it works for every path
above and shows the data through the input normalizers.

With ``wrap_raw: false`` neuroglancer reads the data in the browser, so the
browser must be able to reach it. Its source is the dataset's URL with the
format in front: ``zarr://s3://…``, ``zarr://gs://…``, ``zarr://https://…``,
``n5://…``, or ``precomputed://gs://…`` (a precomputed volume, never one of
its scales). The browser has none of the credentials above and cannot open
local files, so a private bucket or a local path needs ``wrap_raw: true``.
