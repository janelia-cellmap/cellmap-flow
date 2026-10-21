"""Where a dataset's files are, and reading them: worldarray's kvstore access.

A location is a local directory or an ``http(s)://``, ``s3://`` or ``gs://``
URL. worldarray (``worldarray.store``) gives the tensorstore kvstore specs that
read it and moves to the next only when a read is refused (s3: anonymous, then
the AWS default chain, ``AWS_ENDPOINT_URL(_S3)`` and ``AWS_REGION`` honoured;
gs: Google's default credentials, then the bucket's public https URL, which
``TENSORSTORE_GCS_HTTP_URL`` moves; GCE's metadata server is not probed); the
way that worked is remembered per bucket. This module keeps cellmap-flow's
location-first reads over it.
"""

from typing import Optional

from worldarray.store import Location


def read(location: str, key: str) -> Optional[bytes]:
    """The file ``key`` in the directory ``location``; None when it is not there."""
    return Location(location).read(key)


def read_json(location: str, key: str) -> Optional[dict]:
    """The JSON document ``key`` in the directory ``location``; None when it
    is not there."""
    return Location(location).read_json(key)


def exists(location: str, key: str) -> bool:
    return read(location, key) is not None
