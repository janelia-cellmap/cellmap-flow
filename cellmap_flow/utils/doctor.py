"""
Preflight check for the cellmap-flow environment.

Run as ``cellmap_flow_doctor``. Reports what is installed, what is missing,
and the exact command to fix each gap -- so a broken finetune environment
surfaces up front rather than as a FileNotFoundError deep in an annotation
route after you have already loaded a dataset and started painting.
"""

import importlib
import importlib.util
import shutil
import sys
from pathlib import Path

OK = "✓"
FAIL = "✗"
WARN = "!"

# Distinctive string from the voxel-annotation fork's Draw-tab controls
# (src/layer/voxel_annotation/controls.ts). Upstream neuroglancer does not
# ship it, so its presence in the bundled client is a reliable marker that
# the painting tools -- and the S3 kvstore write support they depend on --
# are actually built into the installed package. Checking the version number
# would not work: the fork keeps upstream's version string.
NEUROGLANCER_FORK_MARKER = "Paint Value"

NEUROGLANCER_FORK_INSTALL = (
    "pip install git+https://github.com/briossant/neuroglancer@feature/voxel-annotation"
)

# MinIO stopped publishing prebuilt community server binaries: dl.min.io
# returns 410 Gone, and the minio/minio GitHub releases carry no assets.
# conda-forge still builds it from source, so it is the only practical source.
MINIO_INSTALL = "mamba install minio-server minio-client -c conda-forge"


class Result:
    def __init__(self, name, status, detail="", fix=""):
        self.name = name
        self.status = status  # "ok" | "fail" | "warn"
        self.detail = detail
        self.fix = fix


def _check_python():
    v = sys.version_info
    version = f"{v.major}.{v.minor}.{v.micro}"
    if (v.major, v.minor) < (3, 11):
        return Result(
            "python",
            "fail",
            f"{version} (need >= 3.11)",
            "mamba create -n cellmap-flow-finetune python=3.11",
        )
    return Result("python", "ok", version)


def _check_module(module, label=None, fix="", optional=False):
    label = label or module
    if importlib.util.find_spec(module) is None:
        return Result(label, "warn" if optional else "fail", "not installed", fix)
    try:
        importlib.import_module(module)
    except Exception as e:
        return Result(label, "fail", f"installed but failed to import: {e}", fix)

    # Prefer package metadata over __version__: Flask deprecated the attribute,
    # and submodules (google.genai) do not carry one at all.
    try:
        from importlib.metadata import version

        detail = version(label)
    except Exception:
        detail = ""
    return Result(label, "ok", detail)


def _check_torch_cuda():
    if importlib.util.find_spec("torch") is None:
        return Result("torch cuda", "fail", "torch not installed", 'pip install -e ".[finetune]"')
    import torch

    if not torch.cuda.is_available():
        return Result(
            "torch cuda",
            "warn",
            f"torch {torch.__version__} installed but CUDA unavailable -- training will use CPU",
            "Check the torch build matches the driver's CUDA version "
            "(pip wheels fall back to CPU silently when it does not).",
        )
    return Result(
        "torch cuda", "ok", f"torch {torch.__version__}, {torch.cuda.device_count()} device(s)"
    )


def _check_neuroglancer_fork():
    """Verify the installed neuroglancer actually has the voxel-annotation tools."""
    spec = importlib.util.find_spec("neuroglancer")
    if spec is None or not spec.origin:
        return Result("neuroglancer fork", "fail", "not installed", NEUROGLANCER_FORK_INSTALL)

    static_dir = Path(spec.origin).parent / "static" / "client"
    if not static_dir.is_dir():
        return Result(
            "neuroglancer fork",
            "fail",
            "no bundled client found -- cannot verify annotation support",
            NEUROGLANCER_FORK_INSTALL,
        )

    for js in static_dir.glob("*.js"):
        try:
            if NEUROGLANCER_FORK_MARKER in js.read_text(errors="ignore"):
                return Result("neuroglancer fork", "ok", "voxel-annotation tools present")
        except OSError:
            continue

    return Result(
        "neuroglancer fork",
        "fail",
        "upstream neuroglancer installed -- painting tools missing",
        NEUROGLANCER_FORK_INSTALL,
    )


def _check_binary(name, fix):
    path = shutil.which(name)
    if path is None:
        return Result(name, "fail", "not on PATH", fix)
    return Result(name, "ok", path)


def run_checks(include_finetune=True):
    results = [_check_python()]

    for module in ["torch", "zarr", "tensorstore", "neuroglancer", "flask"]:
        results.append(_check_module(module, fix="pip install -e ."))

    results.append(_check_torch_cuda())

    if not include_finetune:
        return results

    results.append(_check_neuroglancer_fork())

    for module in ["peft", "transformers", "accelerate"]:
        results.append(_check_module(module, fix='pip install -e ".[finetune]"'))

    results.append(
        _check_module(
            "google.genai", "google-genai", fix='pip install -e ".[finetune]"', optional=True
        )
    )

    results.append(_check_binary("minio", MINIO_INSTALL))
    results.append(_check_binary("mc", MINIO_INSTALL))

    return results


def main():
    include_finetune = "--core-only" not in sys.argv
    results = run_checks(include_finetune=include_finetune)

    width = max(len(r.name) for r in results) + 2
    symbols = {"ok": OK, "fail": FAIL, "warn": WARN}

    print("\ncellmap-flow environment check\n")
    for r in results:
        print(f"  {symbols[r.status]} {r.name.ljust(width)} {r.detail}")

    problems = [r for r in results if r.status != "ok" and r.fix]
    if problems:
        print("\nTo fix:\n")
        seen = set()
        for r in problems:
            if r.fix not in seen:
                seen.add(r.fix)
                print(f"  {r.fix}")

    failures = [r for r in results if r.status == "fail"]
    if failures:
        print(f"\n{len(failures)} blocking problem(s).\n")
        return 1

    warnings = [r for r in results if r.status == "warn"]
    print(f"\nEnvironment OK{f' ({len(warnings)} warning(s))' if warnings else ''}.\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
