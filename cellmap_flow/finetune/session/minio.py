"""The dashboard's MinIO: the local S3 server annotation volumes are painted through.

Neuroglancer reads and writes a volume's chunks straight from MinIO; the
volume on disk is mirrored up when it is served and pulled back by
``session.sync``. The server's state -- process, address, bucket, data
directory, sync thread -- lives in the dict a ``MinioServer`` is given (the
dashboard's ``minio_state``), never here, so any number of them over the
same dict agree.
"""

import logging
import os
import socket
import subprocess
import threading
import time
from pathlib import Path
from typing import Optional
from urllib.parse import urlparse

import s3fs

logger = logging.getLogger(__name__)

# The mc alias for this dashboard's MinIO, defined per call through
# MC_HOST_<alias> (see MinioServer.mc_env) rather than with `mc alias set`,
# whose ~/.mc/config.json every dashboard of the user shares.
# finetune_utils.MC_ALIAS repeats it; those lines are pinned for the
# fileglancer branch.
MC_ALIAS = "myserver"
MINIO_READY_TIMEOUT = 30.0
_CREDENTIALS = ("minio", "minio123")

# The URL a reverse proxy in front of the dashboard serves this dashboard's
# MinIO at, for browsers that reach the dashboard through that proxy. Unset,
# MinIO URLs are handed out as they are. "{proto}" and "{host}" stand for the
# forwarded scheme and host, e.g. "{proto}://{host}/minio".
MINIO_PROXY_URL_ENV = "CELLMAP_FLOW_MINIO_PROXY_URL"

# Serializes starting MinIO: two requests arriving together could each see
# no server and start one.
_minio_lock = threading.Lock()


def get_local_ip():
    """Get the local IP address for MinIO server."""
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        local_ip = s.getsockname()[0]
        s.close()
        return local_ip
    except Exception:
        return "127.0.0.1"


def find_available_port(start_port=9000):
    """Find an available port pair for MinIO server (API on port, console on port+1)."""
    for port in range(start_port, start_port + 100):
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s1:
                s1.bind(("", port))
                with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s2:
                    s2.bind(("", port + 1))
                    return port
        except OSError:
            continue
    raise RuntimeError("Could not find available port for MinIO")


def _mc_env(ip, port) -> dict:
    user, password = _CREDENTIALS
    env = os.environ.copy()
    env[f"MC_HOST_{MC_ALIAS}"] = f"http://{user}:{password}@{ip}:{port}"
    return env


def minio_root(output_base_dir) -> Path:
    """Where MinIO started for ``output_base_dir`` keeps its data."""
    if output_base_dir:
        return Path(output_base_dir) / ".minio"
    return Path("~/.minio-server").expanduser()


def wait_for_ready(ip, port, process, timeout=MINIO_READY_TIMEOUT) -> bool:
    """Poll MinIO's readiness endpoint until it answers 200, the process exits, or time runs out."""
    import urllib.request

    url = f"http://{ip}:{port}/minio/health/ready"
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if process.poll() is not None:
            return False
        try:
            with urllib.request.urlopen(url, timeout=1) as response:
                if response.status == 200:
                    return True
        except Exception:
            pass
        time.sleep(0.2)
    return False


def make_s3_filesystem(state):
    """An s3fs filesystem pointed at the MinIO ``state`` describes.

    Both cache opt-outs are load-bearing, not tuning knobs.

    fsspec caches filesystem *instances* keyed on their constructor
    arguments, so every call here would otherwise hand back the same object
    -- and with it the same ``dircache``. s3fs fills ``dircache`` on ``ls()``
    and never expires it by default. The periodic sync thread starts when the
    annotation volume is created, so its first listing of ``annotation/s0``
    runs before the user has painted anything and caches a chunk-less
    listing. Every later sync then reuses that stale listing,
    ``diff_and_sync_chunks`` sees no chunk keys, and painted scribbles never
    reach disk -- while ``sync_zarr_group_metadata`` keeps working, because
    ``cat()``/``exists()`` address objects directly and bypass the cache.
    The symptom is a permanent "Synced 0/N annotations" and a training run
    that dies with "No corrections found".

    Listings here are small and served by a local MinIO, so not caching them
    costs nothing.
    """
    key, secret = _CREDENTIALS
    return s3fs.S3FileSystem(
        anon=False,
        key=key,
        secret=secret,
        client_kwargs={
            "endpoint_url": f"http://{state['ip']}:{state['port']}",
            "region_name": "us-east-1",
        },
        skip_instance_cache=True,
        use_listings_cache=False,
    )


def proxied_url(minio_url, request=None):
    """``minio_url`` as a browser behind a reverse proxy can reach it.

    A browser that loaded the dashboard over a proxy's HTTPS cannot fetch
    ``http://<node>:<port>/...`` chunks from MinIO: the node may not be
    reachable, and the page blocks the mixed content. When
    CELLMAP_FLOW_MINIO_PROXY_URL is set and ``request`` (anything with
    ``headers`` and ``scheme``) came through a proxy, the URL's path is put
    under that URL instead.

    Only a request carrying X-Forwarded-Host came through a proxy. Falling
    back to the Host header rewrote the URL for every browser reaching the
    dashboard directly. The scheme is X-Forwarded-Proto's, else the
    request's own. With no request the URL is returned unchanged.
    """
    template = os.environ.get(MINIO_PROXY_URL_ENV, "").strip()
    if not template or request is None:
        return minio_url
    forwarded_host = request.headers.get("X-Forwarded-Host")
    if not forwarded_host:
        return minio_url
    # Each proxy in a chain appends its own; the first is the client's.
    host = forwarded_host.split(",")[0].strip()
    proto = (request.headers.get("X-Forwarded-Proto") or request.scheme).split(",")[0].strip()
    base = template.replace("{proto}", proto).replace("{host}", host).rstrip("/")
    return base + urlparse(minio_url).path


class MinioServer:
    """Starts MinIO once, then serves annotation volumes through it.

    ``state`` is the dashboard's ``minio_state`` dict (``process``, ``ip``,
    ``port``, ``bucket``, ``output_base``, ``log_path``, ``sync_thread``):
    it is read and written in place. ``volumes`` is the volume registry the
    sync keeps its per-chunk state in. ``preflight``, if given, runs before
    anything else, e.g. a check that the binaries are installed.
    """

    def __init__(self, state: dict, preflight=None, volumes: Optional[dict] = None):
        self.state = state
        self.preflight = preflight
        self.volumes = {} if volumes is None else volumes

    def running(self) -> bool:
        process = self.state.get("process")
        return process is not None and process.poll() is None

    def mc_env(self) -> dict:
        """The environment for an `mc` call that addresses this server as MC_ALIAS."""
        return _mc_env(self.state["ip"], self.state["port"])

    def url_for(self, zarr_name: str) -> str:
        """The URL a browser reads ``<bucket>/<zarr_name>`` at, before any proxy."""
        return f"http://{self.state['ip']}:{self.state['port']}/{self.state['bucket']}/{zarr_name}"

    def ensure_serving(self, zarr_path: str, volume_id: str, output_base_dir: Optional[str] = None,
                       mc_target_name: Optional[str] = None) -> str:
        """Start MinIO if needed, then upload the volume at ``zarr_path``.

        Args:
            zarr_path: The volume to serve.
            volume_id: Its id, which the sync finds its chunks by: the bucket
                key without ".zarr".
            output_base_dir: Where MinIO keeps its data if it has to start
                (``<output_base_dir>/.minio``); a running one keeps its own.
            mc_target_name: The bucket key, when it is not
                ``basename(zarr_path)``: instance corrections keep one key per
                ROI whichever snapshot on disk is served.

        Returns:
            The volume's MinIO URL.
        """
        if self.preflight is not None:
            self.preflight()
        with _minio_lock:
            if not self.running():
                self._start(output_base_dir)

        # `mc mirror --overwrite` pushes every local chunk over MinIO's copy, so
        # anything painted since the last sync -- up to 30 s of strokes, or all
        # of them for a resumed session whose .minio holds strokes never synced
        # -- would be overwritten by a stale local chunk. Pull those first.
        self._pull_painted_chunks(zarr_path, volume_id, mc_target_name)

        zarr_name = mc_target_name or Path(zarr_path).name
        target = f"{MC_ALIAS}/{self.state['bucket']}/{zarr_name}"
        logger.info(f"Uploading {zarr_name} to MinIO")
        result = subprocess.run(
            ["mc", "mirror", "--overwrite", zarr_path, target],
            capture_output=True,
            text=True,
            env=self.mc_env(),
        )
        if result.returncode != 0:
            raise RuntimeError(f"Failed to upload to MinIO: {result.stderr}")
        logger.info(f"Uploaded {zarr_name} to MinIO")
        return self.url_for(zarr_name)

    def _start(self, output_base_dir):
        """Start MinIO and set it up; record it in the state only once all of that worked.

        The state used to be recorded right after the process started, before
        the bucket and policy were set up, so a failure in any of those left a
        live process that later calls took for a working server -- no bucket
        and no sync thread until the dashboard restarted. Its log goes to a
        file beside its data directory: a pipe nobody reads blocks MinIO once
        it has logged about 64 KB.
        """
        from cellmap_flow.finetune.session import sync

        root = minio_root(output_base_dir)
        root.mkdir(parents=True, exist_ok=True)
        log_path = root.parent / f"{root.name}.log"

        ip = get_local_ip()
        port = find_available_port()
        user, password = _CREDENTIALS
        env = os.environ.copy()
        env["MINIO_ROOT_USER"] = user
        env["MINIO_ROOT_PASSWORD"] = password
        env["MINIO_API_CORS_ALLOW_ORIGIN"] = "*"
        minio_cmd = [
            "minio", "server", str(root),
            "--address", f"{ip}:{port}",
            "--console-address", f"{ip}:{port+1}",
        ]

        logger.info(f"Starting MinIO server at {ip}:{port} (log: {log_path})")
        with open(log_path, "ab") as log:
            process = subprocess.Popen(minio_cmd, env=env, stdout=log, stderr=subprocess.STDOUT)

        try:
            if not wait_for_ready(ip, port, process):
                raise RuntimeError(
                    f"MinIO did not become ready at {ip}:{port} within "
                    f"{MINIO_READY_TIMEOUT:.0f}s; see {log_path}"
                )
            mc_env = _mc_env(ip, port)
            bucket = f"{MC_ALIAS}/{self.state['bucket']}"
            result = subprocess.run(["mc", "mb", bucket], capture_output=True, text=True, env=mc_env)
            if result.returncode != 0 and "already" not in result.stderr.lower():
                raise RuntimeError(f"Could not create the MinIO bucket {bucket}: {result.stderr}")
            subprocess.run(
                ["mc", "anonymous", "set", "public", bucket],
                check=True,
                capture_output=True,
                env=mc_env,
            )
        except Exception:
            process.terminate()
            try:
                process.wait(timeout=10)
            except Exception:
                process.kill()
            raise

        self.state["output_base"] = output_base_dir if output_base_dir else None
        self.state["log_path"] = str(log_path)
        self.state["port"] = port
        self.state["ip"] = ip
        self.state["process"] = process
        logger.info(f"MinIO started (PID: {process.pid})")
        sync.start_periodic_sync(self.state, self.volumes)

    def _pull_painted_chunks(self, zarr_path, volume_id, mc_target_name=None):
        """Sync the volume's chunks from MinIO into ``zarr_path``, if MinIO has any.

        A fresh volume has nothing in the bucket yet; that costs one request.
        """
        from cellmap_flow.finetune.session import sync

        zarr_name = mc_target_name or Path(zarr_path).name
        try:
            s3 = make_s3_filesystem(self.state)
            if not s3.exists(f"{self.state['bucket']}/{zarr_name}/annotation/s0"):
                return
        except Exception as e:
            logger.warning(f"Could not check MinIO for painted chunks of {zarr_name}: {e}")
            return
        sync.sync_volume(volume_id, zarr_path=str(zarr_path), state=self.state, volumes=self.volumes)
