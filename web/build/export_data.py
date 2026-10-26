#!/usr/bin/env python
"""Export the constants and catalogs the browser app shares with Python.

The browser mode re-implements parts of cellmap_flow in TypeScript, and those
parts lean on data: which normalizers and postprocessors the browser runs and
their parameters, the chain a fresh session starts with, the organelle
catalog and the prompt's sentences, the viewer's shaders and colours, the
ranking's constants, the finetune log's patterns, the unit table, the
version a model published from the browser names, the query parameters of
the app's links and the Hugging Face sign-in's client id and scopes. This
script reads them from the Python that owns them and writes them to
web/src/generated/python-data.json, which web/src/generated/python-data.ts
gives to the app. One source, so the two modes cannot drift apart.

The file is committed: the typecheck and the vitest suite read it without
Python. ``npm run export:data`` and ``npm run render`` regenerate it, and
tests/web/test_python_data_fresh.py fails when it is stale. It is written in
Python's own dict order, because some of it carries meaning in its order (a
chain step's keys are the bytes of its layer's URL, and the organelles are
listed in the catalog's order). Sets are sorted, and nothing in it changes
from one run to the next.

Usage (from web/)::

    python build/export_data.py            # writes src/generated/python-data.json
    python build/export_data.py --check    # exits 1 if it is stale

Exports the cellmap_flow beside this script (the repository root, two
directories up, put first on sys.path), as build/render.py renders it. Its
dependencies must be importable (the pixi ``default`` environment has them).
"""
import argparse
import json
import re
import sys
from dataclasses import asdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

WEB = Path(__file__).resolve().parents[1]
OUT = WEB / "src" / "generated" / "python-data.json"

# The chains a fresh browser session starts with, ticked on the Input and
# Postprocess tabs with these values (build/render.py) and served by the app
# (web/src/state/index.ts). The six models in Hugging Face's
# cellmap/onnx-web-models were trained on uint8 raw scaled to [-1, 1] and
# output logits, so this is what every one of them wants: anything else gives
# wrong predictions without a hint of why. (The Flask dashboard starts with
# empty chains because its models vary.)
DEFAULT_CHAIN = {
    "input_norm": [
        {"name": "MinMaxNormalizer", "min_value": 0, "max_value": 255},
        {"name": "LambdaNormalizer", "expression": "2*x-1"},
    ],
    "postprocess": [{"name": "SigmoidPostprocessor"}],
}

_JS_FLAGS = ((re.IGNORECASE, "i"), (re.MULTILINE, "m"), (re.DOTALL, "s"))


def regex(pattern):
    """A compiled pattern as JavaScript's RegExp takes it: ``{source, flags}``.

    The patterns exported are written in the syntax the two languages share
    (``\\s``, ``\\d``, classes, groups); a flag JavaScript has no twin of is
    refused rather than dropped.
    """
    known = re.UNICODE
    for flag, _ in _JS_FLAGS:
        known |= flag
    if pattern.flags & ~known:
        raise ValueError(f"{pattern.pattern!r} has a flag JavaScript's RegExp lacks")
    return {"source": pattern.pattern, "flags": "".join(js for flag, js in _JS_FLAGS if pattern.flags & flag)}


def ops(listing):
    """A registry (input_normalize.op_listing) as the export holds it: each
    op's name, whether the browser runs it, and its parameters' defaults."""
    return [{"name": op["name"], "browser": op["browser"], "params": op["params"]} for op in listing]


def python_data():
    """Everything the export holds, as one JSON-ready dict.

    A few of the tables are module-private (``_ALIASES``, ``_COLOR_RE``,
    ``_NM_PER_UNIT``): read here by name, so renaming one fails this export
    and its freshness test.
    """
    import cellmap_flow
    from cellmap_flow.ai_annotate import config as ai_config
    from cellmap_flow.ai_annotate import geometry as ai_geometry
    from cellmap_flow.ai_annotate import images as ai_images
    from cellmap_flow.ai_annotate import organelles, prompts
    from cellmap_flow.dashboard.routes.finetune import ai_annotate as ai_routes
    from cellmap_flow.dashboard.routes.finetune import view_labels
    from cellmap_flow.finetune import markers
    from cellmap_flow.portable import publish, rank
    from cellmap_flow.finetune.job_manager.state import TERMINAL_STATUSES, JobStatus
    from cellmap_flow.io import metadata
    from cellmap_flow.norm.input_normalize import get_input_normalizers
    from cellmap_flow.post.postprocessors import get_postprocessors_list
    from cellmap_flow.viewer import layers, raw

    return {
        # The release, which a model published to Hugging Face from the
        # browser names in its card and manifest (web/src/publish/model.ts
        # hands it to cellmap_flow/portable/publish.py).
        "version": cellmap_flow.__version__,
        # The query parameters of the app's links, which a published model's
        # card writes (portable/publish.py) and web/src/link.ts reads back.
        "links": {"model_param": publish.MODEL_PARAM, "dataset_param": publish.DATASET_PARAM},
        # Sign in with Hugging Face (web/src/publish/oauth.ts), as the
        # server's sign-in asks for it too.
        "huggingface": {
            "oauth_client_id": publish.HF_OAUTH_CLIENT_ID,
            "oauth_scopes": list(publish.HF_OAUTH_SCOPES),
            # What both modes' publish routes answer in fixed words.
            "sign_in_first": publish.SIGN_IN_FIRST,
            "token_refused": publish.TOKEN_REFUSED,
            "read_only": publish.READ_ONLY,
        },
        "ops": {
            "input_norm": ops(get_input_normalizers()),
            "postprocess": ops(get_postprocessors_list()),
        },
        "default_chain": DEFAULT_CHAIN,
        "ai_annotate": {
            "organelles": [asdict(profile) for profile in organelles.ORGANELLES.values()],
            "aliases": organelles._ALIASES,
            "default_color_name": organelles._DEFAULT_COLOR_NAME,
            "default_rgb": list(organelles._DEFAULT_RGB),
            "sentences": prompts.SENTENCES,
            "max_prompt_chars": prompts.MAX_PROMPT_CHARS,
            "daily_call_limit": ai_config.DEFAULT_DAILY_CALL_LIMIT,
            "crop_size_px": ai_config.DEFAULT_CROP_SIZE_PX,
            "timeout_s": ai_config.DEFAULT_TIMEOUT_S,
            "max_image_side": ai_geometry.MAX_IMAGE_SIDE,
            "max_image_pixels": ai_images.MAX_PIXELS,
            "stage_labels": ai_routes.STAGE_LABELS,
            "key": ai_routes.KEY,
            "action": ai_routes.ACTION,
            "keybinding": ai_routes.KEYBINDING,
        },
        "viewer": {
            "raw_shader": raw.RAW_SHADER,
            "prediction_shader": raw.PREDICTION_SHADER,
            "prediction_colors": raw.PREDICTION_COLORS,
            "color_control": regex(layers._COLOR_RE),
            "source_bound_layer_keys": sorted(layers.SOURCE_BOUND_LAYER_KEYS),
        },
        "finetune": {
            "rank_blocks": rank.RANK_BLOCKS,
            "rank_smoothing": rank.RANK_SMOOTHING,
            "narrow_window": rank.NARROW_WINDOW,
            "window_margin": rank.WINDOW_MARGIN,
            "cutoffs": rank.CUTOFFS.tolist(),
            "job_statuses": [status.value for status in JobStatus],
            "terminal_statuses": sorted(status.value for status in TERMINAL_STATUSES),
            "epoch_start": regex(markers.EPOCH_START_RE),
            "epoch_summary": regex(markers.EPOCH_SUMMARY_RE),
            "restarting_marker": markers.RESTARTING_TRAINING,
            # The Label it tools' (view_labels.py).
            "undo_depth": view_labels.UNDO_DEPTH,
            "large_box_voxels": view_labels.LARGE_BOX_VOXELS,
        },
        "units": {"nm_per_unit": metadata._NM_PER_UNIT},
    }


def render_json():
    """The export's text, as python-data.json holds it."""
    return json.dumps(python_data(), indent=2, ensure_ascii=False) + "\n"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", default=str(OUT), help="where to write it (default: web/src/generated/python-data.json)")
    parser.add_argument("--check", action="store_true", help="write nothing; exit 1 if --out does not hold the export")
    args = parser.parse_args(argv)

    text = render_json()
    out = Path(args.out)
    current = out.read_text(encoding="utf-8") if out.exists() else None
    if args.check:
        if current == text:
            print(f"{out} is up to date")
            return 0
        print(f"{out} is {'missing' if current is None else 'stale'}: run build/export_data.py", file=sys.stderr)
        return 1
    if current != text:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text, encoding="utf-8")
    print(f"exported cellmap_flow's shared data -> {out} ({len(text):,} chars)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
