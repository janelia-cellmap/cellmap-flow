#!/usr/bin/env python3
"""
Probe cross-node file visibility delay on a shared filesystem.

Typical usage on two hosts:

1) On watcher host (node A), start waiting:
   python scripts/fs_visibility_probe.py watch \
       --dir /shared/path/probe \
       --token run1 \
       --interval 0.2

2) On writer host (node B), create marker:
   python scripts/fs_visibility_probe.py write \
       --dir /shared/path/probe \
       --token run1

Watcher prints:
- detect_elapsed_s: local watcher elapsed time since it started waiting
- mtime_to_detect_s: local watcher now - marker file mtime (good signal, no cross-host app clock dependency)
- writer_timestamp_to_detect_s: local watcher now - writer timestamp from file (can be skewed by host clock drift)
"""

from __future__ import annotations

import argparse
import json
import os
import socket
import time
from datetime import datetime
from pathlib import Path


def _iso_now() -> str:
    return datetime.now().isoformat()


def _marker_path(base_dir: Path, token: str) -> Path:
    return base_dir / f"fs_probe_{token}.json"


def cmd_write(base_dir: Path, token: str, overwrite: bool) -> int:
    base_dir.mkdir(parents=True, exist_ok=True)
    marker = _marker_path(base_dir, token)
    if marker.exists() and not overwrite:
        print(f"ERROR marker already exists: {marker}")
        print("Use --overwrite or choose a new --token.")
        return 2

    payload = {
        "token": token,
        "writer_host": socket.gethostname(),
        "writer_pid": os.getpid(),
        "writer_iso": _iso_now(),
        "writer_epoch": time.time(),
    }

    with open(marker, "w") as f:
        json.dump(payload, f, indent=2)
        f.flush()
        # Best effort: push data to server before returning.
        os.fsync(f.fileno())

    stat = marker.stat()
    print(f"wrote={marker}")
    print(f"writer_host={payload['writer_host']}")
    print(f"writer_epoch={payload['writer_epoch']:.6f}")
    print(f"marker_mtime_epoch={stat.st_mtime:.6f}")
    print(f"marker_mtime_iso={datetime.fromtimestamp(stat.st_mtime).isoformat()}")
    return 0


def cmd_watch(base_dir: Path, token: str, interval: float, timeout: float) -> int:
    marker = _marker_path(base_dir, token)
    watch_host = socket.gethostname()
    start_perf = time.perf_counter()
    start_epoch = time.time()
    start_iso = _iso_now()
    polls = 0
    next_diag = start_perf + 10.0

    print(f"watching={marker}")
    print(f"watch_host={watch_host}")
    print(f"watch_start_iso={start_iso}")
    print(f"watch_start_epoch={start_epoch:.6f}")
    print(f"interval_s={interval}")
    print(f"timeout_s={timeout}")

    while True:
        now_perf = time.perf_counter()
        now_epoch = time.time()
        elapsed = now_perf - start_perf
        polls += 1

        if marker.exists():
            stat = marker.stat()
            try:
                payload = json.loads(marker.read_text())
            except Exception as e:
                print(f"ERROR reading marker JSON: {e}")
                return 3

            mtime_to_detect = now_epoch - stat.st_mtime
            writer_epoch = payload.get("writer_epoch")
            writer_to_detect = (now_epoch - float(writer_epoch)) if writer_epoch is not None else None

            print("detected=1")
            print(f"detect_elapsed_s={elapsed:.6f}")
            print(f"polls={polls}")
            print(f"detect_iso={_iso_now()}")
            print(f"marker_mtime_epoch={stat.st_mtime:.6f}")
            print(f"marker_mtime_iso={datetime.fromtimestamp(stat.st_mtime).isoformat()}")
            print(f"mtime_to_detect_s={mtime_to_detect:.6f}")
            if writer_to_detect is not None:
                print(f"writer_timestamp_to_detect_s={writer_to_detect:.6f}")
                print(f"writer_host={payload.get('writer_host')}")
                print(f"writer_iso={payload.get('writer_iso')}")
            return 0

        if elapsed >= timeout:
            print("detected=0")
            print(f"timeout_after_s={elapsed:.6f}")
            print(f"polls={polls}")
            return 1

        if now_perf >= next_diag:
            print(f"waiting elapsed_s={elapsed:.2f} polls={polls}")
            next_diag += 10.0

        time.sleep(interval)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Shared filesystem visibility probe")
    sub = parser.add_subparsers(dest="command", required=True)

    p_write = sub.add_parser("write", help="Create probe marker file")
    p_write.add_argument("--dir", required=True, type=Path, help="Shared directory")
    p_write.add_argument("--token", required=True, help="Probe token")
    p_write.add_argument("--overwrite", action="store_true", help="Overwrite marker if it exists")

    p_watch = sub.add_parser("watch", help="Poll for probe marker file visibility")
    p_watch.add_argument("--dir", required=True, type=Path, help="Shared directory")
    p_watch.add_argument("--token", required=True, help="Probe token")
    p_watch.add_argument("--interval", type=float, default=0.2, help="Poll interval in seconds")
    p_watch.add_argument("--timeout", type=float, default=180.0, help="Timeout in seconds")

    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "write":
        return cmd_write(args.dir, args.token, args.overwrite)
    if args.command == "watch":
        return cmd_watch(args.dir, args.token, args.interval, args.timeout)

    parser.error(f"Unknown command: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
