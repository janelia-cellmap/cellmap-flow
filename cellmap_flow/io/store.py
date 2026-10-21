"""Where a dataset's files are, and reading them: worldarray's kvstore access.

A location is a local directory or an ``http(s)://``, ``s3://`` or ``gs://``
URL. worldarray (``worldarray.store``) gives the tensorstore kvstore specs that
read it, in the order to try them, and moves to the next only when a read is
refused (s3: anonymous, then the AWS default chain; gs: Google's default
credentials, then the bucket's public https URL, which
``TENSORSTORE_GCS_HTTP_URL`` moves); the way that worked is remembered per
bucket. This module keeps cellmap-flow's location-first signatures over it.
"""

import os
from typing import Callable, List, Optional, TypeVar

from worldarray.store import GCS_PUBLIC_URL, Location, kvstore_specs  # noqa: F401
from worldarray.store import with_access as _with_access

T = TypeVar("T")

URL_SCHEMES = ("http://", "https://", "s3://", "gs://")


def is_url(location: str) -> bool:
    return location.startswith(URL_SCHEMES)


def prepare(location: str) -> str:
    """``location``, after making sure a ``gs://`` read does not probe GCE's
    metadata server for credentials, which stalls an open off Google Cloud.
    worldarray does not do this; every cellmap-flow entry point into it does."""
    if "gs://" in location:
        os.environ.setdefault("GCE_METADATA_ROOT", "metadata.google.internal.invalid")
    return location


def kvstores(location: str) -> List[dict]:
    """The tensorstore kvstore specs that read ``location``, a directory, in
    the order to try them."""
    return kvstore_specs(prepare(location))


def with_access(location: str, attempt: Callable[[dict], T]) -> T:
    """``attempt(spec)`` with ``location``'s kvstore specs in turn, moving on
    only while access is refused."""
    return _with_access(kvstores(location), attempt)


def read(location: str, key: str) -> Optional[bytes]:
    """The file ``key`` in the directory ``location``; None when it is not there."""
    return Location(prepare(location)).read(key)


def read_json(location: str, key: str) -> Optional[dict]:
    """The JSON document ``key`` in the directory ``location``; None when it
    is not there."""
    return Location(prepare(location)).read_json(key)


def exists(location: str, key: str) -> bool:
    return read(location, key) is not None
