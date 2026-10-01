"""Where a dataset's files are, and reading them: one tensorstore kvstore.

A location is a local directory or an ``http(s)://``, ``s3://`` or ``gs://``
URL. ``io.metadata`` reads the JSON documents there (``.zarray``,
``attributes.json``, ``zarr.json``, a precomputed ``info``) with
``read_json``, and ``io.source`` opens the arrays on the same kvstores
(``with_access``), so the metadata and the voxels come from the same place by
the same route, and a URL is read the way the same files on disk are.

Credentials (docs/source/data_paths.rst) are tried only when a public read is
refused:

- ``s3://``: anonymously first, so a public bucket is read whatever AWS
  credentials are configured (S3 refuses even a public read signed with a
  stale or foreign key); then with the AWS default chain (environment,
  ``~/.aws``, instance metadata). ``AWS_ENDPOINT_URL_S3`` or
  ``AWS_ENDPOINT_URL`` names an S3-compatible endpoint, and ``AWS_REGION``
  or ``AWS_DEFAULT_REGION`` its region.
- ``gs://``: Google application-default credentials if there are any, else
  anonymous (tensorstore's own rule); then the bucket's public https URL,
  for when those credentials are broken. ``TENSORSTORE_GCS_HTTP_URL``, which
  tensorstore reads for another endpoint, moves that URL with it.
- ``http(s)://`` and local paths: no credentials.

Which way worked is remembered per bucket, so only the first read of a
private bucket is refused.
"""

import functools
import json
import logging
import os
from typing import Callable, List, Optional, Tuple, TypeVar

logger = logging.getLogger(__name__)

URL_SCHEMES = ("http://", "https://", "s3://", "gs://")

GCS_PUBLIC_URL = "https://storage.googleapis.com"

T = TypeVar("T")


def is_url(location: str) -> bool:
    return location.startswith(URL_SCHEMES)


def _bucket_and_prefix(url: str) -> Tuple[str, str]:
    """``("bucket", "path/")`` of an ``s3://`` or ``gs://`` URL; the prefix is
    "" at the bucket's root."""
    bucket, _, path = url.split("://", 1)[1].partition("/")
    path = path.strip("/")
    return bucket, path + "/" if path else ""


def _s3(url: str) -> List[dict]:
    bucket, prefix = _bucket_and_prefix(url)
    base = {"driver": "s3", "bucket": bucket, "path": prefix}
    endpoint = os.environ.get("AWS_ENDPOINT_URL_S3") or os.environ.get("AWS_ENDPOINT_URL")
    if endpoint:
        base["endpoint"] = endpoint
    region = os.environ.get("AWS_REGION") or os.environ.get("AWS_DEFAULT_REGION")
    if region:
        base["aws_region"] = region
    return [
        {**base, "aws_credentials": {"type": "anonymous"}},
        {**base, "aws_credentials": {"type": "default"}},
    ]


def _gs(url: str) -> List[dict]:
    bucket, prefix = _bucket_and_prefix(url)
    public = os.environ.get("TENSORSTORE_GCS_HTTP_URL") or GCS_PUBLIC_URL
    return [
        {"driver": "gcs", "bucket": bucket, "path": prefix},
        {"driver": "http", "base_url": public.rstrip("/"), "path": f"/{bucket}/{prefix}"},
    ]


def kvstores(location: str) -> List[dict]:
    """The tensorstore kvstore specs that read ``location``, a directory, in
    the order to try them (see the module docstring). Each spec's path ends
    in "/", so a key read from it is a file in that directory."""
    if location.startswith(("http://", "https://")):
        scheme, _, rest = location.partition("://")
        host, _, path = rest.partition("/")
        path = path.strip("/")
        return [{"driver": "http", "base_url": f"{scheme}://{host}", "path": f"/{path}/" if path else "/"}]
    if location.startswith("s3://"):
        return _s3(location)
    if location.startswith("gs://"):
        # Not GCE's metadata server: probing it for credentials stalls a
        # gs:// open off Google Cloud.
        os.environ.setdefault("GCE_METADATA_ROOT", "metadata.google.internal.invalid")
        return _gs(location)
    return [{"driver": "file", "path": os.path.abspath(location).rstrip("/") + "/"}]


def _scope(location: str) -> str:
    """What one set of credentials covers: an s3 or gs bucket."""
    return location.split("/", 3)[2] if location.startswith(("s3://", "gs://")) else ""


# {(scheme, bucket): the index into kvstores() of the spec that was let in}
_admitted = {}


def _refused(error: Exception) -> bool:
    return str(error).startswith(("PERMISSION_DENIED", "UNAUTHENTICATED"))


def with_access(location: str, attempt: Callable[[dict], T]) -> T:
    """``attempt(spec)`` with ``location``'s kvstore specs in turn, moving on
    only while access is refused; the one let in is tried first next time.
    Any other error, and the last refusal, is raised."""
    specs = kvstores(location)
    key = (location.split("://")[0], _scope(location))
    first = _admitted.get(key, 0)
    order = [first] + [i for i in range(len(specs)) if i != first]
    for index in order[:-1]:
        try:
            result = attempt(specs[index])
        except Exception as e:
            if not _refused(e):
                raise
            logger.info("%s refused a read (%s); trying the next way in", location, str(e).splitlines()[0][:200])
            continue
        _admitted[key] = index
        return result
    result = attempt(specs[order[-1]])
    _admitted[key] = order[-1]
    return result


@functools.lru_cache(maxsize=256)
def _open(spec_json: str, gcs_url: Optional[str]):
    """The opened kvstore of a spec. ``gcs_url`` (TENSORSTORE_GCS_HTTP_URL,
    which tensorstore reads at open) is part of the cache key only."""
    import tensorstore as ts

    return ts.KvStore.open(json.loads(spec_json)).result()


# How a metadata read retries a failed request: about 5 s in all. tensorstore's
# own default, 32 retries of up to 32 s each, kept a misspelt host's dataset
# loading for a quarter of an hour.
_METADATA_RETRIES = {"max_retries": 5, "initial_delay": "0.2s", "max_delay": "2s"}


def _kvstore(spec: dict):
    if spec["driver"] != "file":
        spec = {**spec, "context": {f"{spec['driver']}_request_retries": _METADATA_RETRIES}}
    return _open(json.dumps(spec, sort_keys=True), os.environ.get("TENSORSTORE_GCS_HTTP_URL"))


def read(location: str, key: str) -> Optional[bytes]:
    """The file ``key`` in the directory ``location``; None when it is not
    there."""

    def attempt(spec):
        result = _kvstore(spec).read(key).result()
        return bytes(result.value) if result.state == "value" else None

    return with_access(location, attempt)


def read_json(location: str, key: str) -> Optional[dict]:
    """The JSON document ``key`` in the directory ``location``; None when it
    is not there."""
    data = read(location, key)
    return None if data is None else json.loads(data)


def exists(location: str, key: str) -> bool:
    return read(location, key) is not None
