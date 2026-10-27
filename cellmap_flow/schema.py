"""The spec of what the dashboard's two backends share, and its generation for TypeScript.

The dashboard's JavaScript speaks to one of two backends, the Flask routes in
server mode and the TypeScript routes of ``web/src/backend`` in browser mode,
and the two must answer alike. What they agree on is written once, in the
pydantic request models and in this module, and TypeScript is generated from
it, so a change on the Python side reaches the browser's types by itself and
an implementation that drifts fails a test:

- ``python -m cellmap_flow.schema --out web/src/generated --docs docs/api.md``
  (``npm run gen`` in web/) writes ``cellmap-flow.schema.json``, a JSON Schema
  (draft 2020-12), ``cellmap-flow.ts``, its types by json-schema-to-typescript
  with the constants and the route table as values, and the route table as
  Markdown;
- ``--check`` says whether the committed files are current, with a diff when
  not (tests/web/test_schema.py and CI's web job run it).

The schema holds:

- every request model of ``dashboard/requests.py``, and the route files' own
  (``ResolveModel``, ``AddModel``, ``PipelineUpdate``), under ``$defs``; the
  root object maps each route to its body's model ("POST /api/set-data");
- ``x-routes``: every route of the dashboard and where it is answered (a
  ``Route``). The method and path come from the Flask app's ``url_map``, the
  availability from ``AVAILABILITY`` below, and the two must agree: a route
  added without a declaration is an error naming it (``routes()``);
- ``x-constants``: the numbers both modes share (``portable/rank.py``'s,
  ``UNDO_DEPTH``, the AI annotation's caps) and the Finetune tab's defaults
  for a blank field, read from ``finetune/options.py``'s ``TrainingOptions``;
- ``x-steps``: each normalizer's and postprocessor's parameters and defaults
  (a ``StepSpec``), from their registries.
"""

from __future__ import annotations

import argparse
import difflib
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Literal, Optional, Union

from pydantic import BaseModel, Field
from pydantic.json_schema import GenerateJsonSchema, models_json_schema

SCHEMA_DRAFT = "https://json-schema.org/draft/2020-12/schema"
ROOT = Path(__file__).resolve().parents[1]
GENERATED_DIR = ROOT / "web" / "src" / "generated"
DOCS_FILE = ROOT / "docs" / "api.md"
SCHEMA_NAME = "cellmap-flow.schema.json"
TS_NAME = "cellmap-flow.ts"
JSON2TS = ROOT / "web" / "node_modules" / ".bin" / "json2ts"
REGENERATE = "python -m cellmap_flow.schema --out web/src/generated --docs docs/api.md (npm run gen in web/)"

Availability = Literal["both", "stub", "server", "browser"]


class Route(BaseModel):
    """One route of the dashboard and where it is answered: "both", by the
    Flask route and the browser's; "stub", the browser answers the server's
    JSON shape with fixed or empty data; "server", the browser answers
    unavailable; "browser", only the page has it. ``request`` names the
    pydantic model its body is parsed with, where there is one."""

    method: Literal["GET", "POST", "PUT", "DELETE"]
    path: str = Field(description="Flask's spelling of the path: a parameter is <name>.")
    availability: Availability
    request: Optional[str] = None


class StepSpec(BaseModel):
    """A normalizer or postprocessor as the dashboard's forms list it: its
    class, its name, and its constructor's parameters with their defaults
    ("" for one without a default)."""

    class_name: str
    name: str
    params: dict[str, Union[str, int, float, bool, None]]


# Where each route is answered (Route.availability). Every rule of the Flask
# app must be declared here, and every "both", "stub" and "browser" route
# registered in the TypeScript router (tests/web/test_schema.py). The
# browser's "server" answer is router.ts's fallback; the template hides those
# routes' controls under [data-mode="browser"] .server-only.
AVAILABILITY: dict[str, Availability] = {
    # Pages
    "GET /": "server",
    "GET /pipeline-builder": "server",
    # Data and models
    "POST /api/set-data": "both",
    "GET /api/huggingface-models": "both",
    "POST /api/huggingface-models/refresh": "both",
    "GET /api/bioimage-models": "stub",
    "POST /api/bioimage-models/refresh": "stub",
    "POST /api/models": "both",
    "GET /api/models/loaded": "both",
    "POST /api/models/remove": "both",
    "POST /api/models/resolve": "both",
    "POST /api/models/add": "both",
    "GET /api/model_advice": "both",
    "PUT /api/pipeline": "both",
    "POST /api/pipeline/apply": "server",
    "POST /api/process": "server",
    # The browser's merging steps set a layer's equivalences in the page.
    "POST /update/equivalences": "server",
    # Settings and logs
    "GET /api/server-config": "both",
    "POST /api/server-config": "both",
    "GET /api/gpu-queues": "stub",
    "GET /api/job-logs": "stub",
    "GET /api/export-config": "both",
    "GET /api/logs/stream": "server",
    # The pipeline builder
    "GET /api/blockwise-config": "server",
    "POST /api/blockwise-config": "server",
    "POST /api/blockwise/validate": "server",
    "POST /api/blockwise/generate": "server",
    "POST /api/blockwise/precheck": "server",
    "POST /api/blockwise/submit": "server",
    "POST /api/bbx-generator": "server",
    "POST /api/bbx-generator/finalize": "server",
    "GET /api/bbx-generator/status": "server",
    "GET /api/model-config-types": "server",
    "POST /api/create-model-config": "server",
    "GET /api/templates/bbox-json": "server",
    # The Review tab
    "POST /api/review/open": "server",
    "GET /api/review/next": "server",
    "POST /api/review/verdict": "server",
    "POST /api/review/undo": "server",
    "GET /api/review/progress": "server",
    "GET /api/review/show/<int:instance_id>": "server",
    "GET /api/review/pick_stream": "server",
    "GET /api/review/status": "server",
    # The external viewer API (docs/source/viewer_api.rst) and instance correction
    "POST /api/viewer/add-image-layer": "server",
    "POST /api/viewer/add-segmentation-layer": "server",
    "POST /api/viewer/remove-layer": "server",
    "POST /api/viewer/rename-layer": "server",
    "POST /api/viewer/create-instance-correction": "server",
    "POST /api/viewer/sync-instance-correction": "server",
    "POST /api/viewer/cc3d-relabel-annotation": "server",
    # The Finetune tab: its preferences, volumes and viewer
    "GET /api/finetune/user-prefs": "both",
    "POST /api/finetune/user-prefs": "both",
    "GET /api/finetune/models": "both",
    "POST /api/finetune/create-volume": "both",
    "POST /api/finetune/add-to-viewer": "both",
    "POST /api/finetune/list-existing-sessions": "both",
    "POST /api/finetune/load-existing-volume": "both",
    "GET /api/finetune/load-existing-volume-progress": "stub",
    "POST /api/finetune/sync-annotations": "stub",
    "POST /api/finetune/refresh-annotated-regions": "stub",
    "GET /api/finetune/session-volume": "browser",
    "GET /api/finetune/volume/<volume_id>/download": "browser",
    "POST /api/finetune/delete-volume": "browser",
    "POST /api/finetune/load-crops": "server",
    "GET /api/finetune/load-crops-progress": "server",
    "GET /api/finetune/read-yaml": "server",
    # Training jobs
    "POST /api/finetune/submit": "both",
    "GET /api/finetune/jobs": "both",
    "GET /api/finetune/job/<job_id>/status": "both",
    "GET /api/finetune/job/<job_id>/logs": "both",
    "GET /api/finetune/job/<job_id>/logs/stream": "both",
    "POST /api/finetune/job/<job_id>/cancel": "both",
    "POST /api/finetune/job/<job_id>/stop-early": "both",
    "POST /api/finetune/job/<job_id>/restart": "both",
    "GET /api/finetune/model/<name>/download": "browser",
    # Good regions
    "GET /api/finetune/good-regions": "both",
    "POST /api/finetune/good-regions/mark-view": "both",
    "POST /api/finetune/good-regions/delete": "both",
    # Which model to start from
    "GET /api/finetune/rank/candidates": "both",
    "POST /api/finetune/rank/score": "both",
    "POST /api/finetune/rank/use": "both",
    "POST /api/finetune/rank/input-norm": "both",
    # Label it
    "GET /api/finetune/view-labels/sources": "both",
    "POST /api/finetune/view-labels/seed": "both",
    "POST /api/finetune/view-labels/background": "both",
    "POST /api/finetune/view-labels/split": "both",
    "POST /api/finetune/view-labels/undo": "both",
    # AI-assisted annotation
    "GET /api/finetune/ai-annotate/config": "both",
    "POST /api/finetune/ai-annotate/credentials": "browser",
    "POST /api/finetune/ai-annotate/settings": "both",
    "POST /api/finetune/ai-annotate/run": "both",
    "GET /api/finetune/ai-annotate/status": "both",
    "POST /api/finetune/ai-annotate/resend": "both",
    "POST /api/finetune/ai-annotate/accept": "both",
    "POST /api/finetune/ai-annotate/reject": "both",
}

# The model each route parses its body with (requests.parse or check). The
# other POSTs read their bodies by hand.
REQUEST_MODELS: dict[str, str] = {
    "POST /api/set-data": "SetData",
    "POST /api/models": "SubmitModels",
    "POST /api/models/remove": "RemoveModel",
    "POST /api/models/resolve": "ResolveModel",
    "POST /api/models/add": "AddModel",
    "PUT /api/pipeline": "PipelineUpdate",
    "POST /update/equivalences": "Equivalences",
    "POST /api/server-config": "ServerConfigUpdate",
    "POST /api/blockwise-config": "BlockwiseSettings",
    "POST /api/blockwise/validate": "BlockwiseValidate",
    "POST /api/blockwise/generate": "BlockwiseGenerate",
    "POST /api/blockwise/precheck": "BlockwisePrecheck",
    "POST /api/blockwise/submit": "BlockwiseSubmit",
    "POST /api/bbx-generator": "BbxGenerator",
    "POST /api/create-model-config": "CreateModelConfig",
    "POST /api/finetune/create-volume": "CreateVolume",
    "POST /api/finetune/load-crops": "LoadCrops",
    "POST /api/finetune/submit": "FinetuneSubmit",
    "POST /api/finetune/job/<job_id>/restart": "FinetuneRestart",
    "POST /api/finetune/ai-annotate/settings": "AIAnnotateSettings",
    "POST /api/finetune/ai-annotate/run": "AIAnnotateRun",
    "POST /api/finetune/ai-annotate/resend": "AIAnnotateResend",
    "POST /api/finetune/ai-annotate/accept": "AIAnnotateDecision",
    "POST /api/finetune/ai-annotate/reject": "AIAnnotateDecision",
}

# The Finetune tab's fields whose default for a blank value is TrainingOptions'
# (x-constants' FINETUNE_DEFAULTS), which the server form shows and the
# browser's DEFAULT_OPTIONS (web/src/finetune/types.ts) takes where it has
# no reason of its own.
FINETUNE_FORM_FIELDS = (
    "lora_r", "num_epochs", "batch_size", "learning_rate", "loss_type", "margin",
    "label_smoothing", "distillation_lambda", "distillation_scope", "balance_classes", "augment",
)

# What each shared constant is, for the generated TypeScript's comments.
CONSTANT_DOCS = {
    "RANK_BLOCKS": "The most painted blocks one rank score reads, spread over the paint (portable/rank.py).",
    "RANK_SMOOTHING": "The label smoothing of the balanced BCE given beside a rank score.",
    "CUTOFFS": "The logits a rank score is read at, -6 to 6 in quarters; the best by Dice is the model's own cut-off.",
    "NARROW_WINDOW": "A raw narrower than this share of a MinMaxNormalizer's range is scored stretched too.",
    "WINDOW_MARGIN": "How much better the stretched score must be to be kept.",
    "UNDO_DEPTH": "How many Label it actions per volume Undo takes back (view_labels.py).",
    "PROMPT_MAX_CHARS": "The longest AI-annotation prompt the routes take (requests.py).",
    "LABEL_MAX_CHARS": "The longest provider or model id and target name the AI-annotation routes take (requests.py).",
    "FINETUNE_DEFAULTS": "The Finetune tab's form at its defaults: what a blank field trains with (finetune/options.py TrainingOptions).",
}


class _Titled(GenerateJsonSchema):
    """Models keep their titles (the types' names); fields' titles would only
    become one-use type aliases in the generated TypeScript."""

    def field_title_should_be_set(self, schema) -> bool:
        return False

    def default_schema(self, schema):
        # A field typed Any with a default is `{"default": x}`, and json2ts
        # reads the default's type into it (null: an object), where the field
        # takes anything. Only a typed field keeps its default.
        json_schema = super().default_schema(schema)
        return {} if set(json_schema) == {"default"} else json_schema


def request_models() -> dict[str, type[BaseModel]]:
    """Every request model by name: the public pydantic models
    dashboard/requests.py defines, and the route files' own."""
    from cellmap_flow.dashboard import requests
    from cellmap_flow.dashboard.routes.model_resolve import AddModel, ResolveModel
    from cellmap_flow.dashboard.routes.pipeline import BuilderCanvas, PipelineUpdate

    models = {
        name: obj for name, obj in vars(requests).items()
        if isinstance(obj, type) and issubclass(obj, BaseModel)
        and obj.__module__ == requests.__name__ and not name.startswith("_")
    }
    for model in (ResolveModel, AddModel, PipelineUpdate, BuilderCanvas):
        models[model.__name__] = model
    return models


def flask_rules() -> list[tuple[str, str]]:
    """(method, path) for every rule of the dashboard but Flask's own /static."""
    from cellmap_flow.dashboard.app import app

    return [
        (method, rule.rule)
        for rule in app.url_map.iter_rules() if not rule.rule.startswith("/static/")
        for method in sorted(rule.methods - {"HEAD", "OPTIONS"})
    ]


def routes(rules: Optional[list[tuple[str, str]]] = None) -> list[Route]:
    """The route table: ``rules`` (by default the Flask app's) with their
    availability, in AVAILABILITY's order. A rule that AVAILABILITY does not
    declare, or a declaration (other than a browser-only route) the app no
    longer has, is a LookupError naming it."""
    served = {f"{method} {path}" for method, path in (flask_rules() if rules is None else rules)}
    undeclared = sorted(served - set(AVAILABILITY))
    stale = sorted(key for key in set(AVAILABILITY) - served if AVAILABILITY[key] != "browser")
    if undeclared or stale:
        raise LookupError(
            (f"routes of the dashboard that cellmap_flow/schema.py's AVAILABILITY does not declare: {undeclared}; "
             f"say where each is answered (both, stub, server or browser). " if undeclared else "")
            + (f"declared routes the dashboard no longer has: {stale}." if stale else "")
        )
    return [
        Route(method=key.split(" ", 1)[0], path=key.split(" ", 1)[1], availability=availability,
              request=REQUEST_MODELS.get(key))
        for key, availability in AVAILABILITY.items()
    ]


def constants() -> dict:
    from cellmap_flow.dashboard.requests import LABEL_MAX_CHARS, PROMPT_MAX_CHARS
    from cellmap_flow.dashboard.routes.finetune.view_labels import UNDO_DEPTH
    from cellmap_flow.portable import rank
    from cellmap_flow.finetune.options import TrainingOptions

    return {
        "RANK_BLOCKS": rank.RANK_BLOCKS,
        "RANK_SMOOTHING": rank.RANK_SMOOTHING,
        "CUTOFFS": [float(c) for c in rank.CUTOFFS],
        "NARROW_WINDOW": rank.NARROW_WINDOW,
        "WINDOW_MARGIN": rank.WINDOW_MARGIN,
        "UNDO_DEPTH": UNDO_DEPTH,
        "PROMPT_MAX_CHARS": PROMPT_MAX_CHARS,
        "LABEL_MAX_CHARS": LABEL_MAX_CHARS,
        "FINETUNE_DEFAULTS": {name: TrainingOptions.model_fields[name].default for name in FINETUNE_FORM_FIELDS},
    }


def steps() -> dict[str, list[dict]]:
    from cellmap_flow.norm.input_normalize import InputNormalizer, get_input_normalizers
    from cellmap_flow.post.postprocessors import PostProcessor, get_postprocessors_list

    def own(base, listed):
        # The registries list every subclass the process holds; one a test or
        # a script defined for itself is not part of the spec.
        ours = {cls.__name__ for cls in base.__subclasses__() if cls.__module__.startswith("cellmap_flow.")}
        return [StepSpec.model_validate(step).model_dump() for step in listed if step["class_name"] in ours]

    return {
        "input_norm": own(InputNormalizer, get_input_normalizers()),
        "postprocess": own(PostProcessor, get_postprocessors_list()),
    }


def schema() -> dict:
    """The whole schema (see the module's docstring)."""
    models = request_models()
    table = routes()
    missing = sorted({name for name in REQUEST_MODELS.values() if name not in models}
                     | {key for key in REQUEST_MODELS if key not in AVAILABILITY})
    assert not missing, f"REQUEST_MODELS names what the dashboard has not: {missing}"
    _, top = models_json_schema(
        [(model, "validation") for model in (Route, StepSpec, *models.values())], schema_generator=_Titled,
    )
    defs = top["$defs"]
    defs["RequestBodies"] = {
        "title": "RequestBodies",
        "description": "The body each route parses with a pydantic model, by `METHOD path`. The routes not "
                       "listed take no body or read theirs by hand.",
        "type": "object",
        "properties": {f"{r.method} {r.path}": {"$ref": f"#/$defs/{r.request}"} for r in table if r.request},
    }
    step_list = {"type": "array", "items": {"$ref": "#/$defs/StepSpec"}}
    return {
        "$schema": SCHEMA_DRAFT,
        "title": "CellmapFlow",
        "description": "What the cellmap-flow dashboard's two backends share: the routes and where each is "
                       "answered (x-routes), their bodies, the constants both modes use (x-constants) and the "
                       "chain steps' parameters (x-steps); generated by `python -m cellmap_flow.schema`.",
        "type": "object",
        "properties": {
            "requests": {"$ref": "#/$defs/RequestBodies"},
            "routes": {"type": "array", "items": {"$ref": "#/$defs/Route"}},
            "steps": {"type": "object", "properties": {"input_norm": step_list, "postprocess": step_list},
                      "required": ["input_norm", "postprocess"]},
        },
        "x-routes": [route.model_dump() for route in table],
        "x-constants": constants(),
        "x-steps": steps(),
        "$defs": dict(sorted(defs.items())),
    }


def dumps(spec: Optional[dict] = None) -> str:
    return json.dumps(schema() if spec is None else spec, indent=2) + "\n"


# --- TypeScript ---------------------------------------------------------------------

BANNER = "/* Generated from cellmap-flow's JSON Schema (python -m cellmap_flow.schema, npm run gen). Do not edit. */"


def typescript(spec: Optional[dict] = None, json2ts: Path = JSON2TS) -> str:
    """``spec``'s types, by json-schema-to-typescript (web's devDependency), then
    its constants, route table and steps as values."""
    spec = schema() if spec is None else spec
    node = shutil.which("node")
    if not node or not json2ts.exists():
        raise FileNotFoundError(f"{json2ts} needs node on PATH and `npm ci` run in web/")
    result = subprocess.run(
        [node, str(json2ts), "--no-additionalProperties", "--bannerComment", BANNER],
        input=json.dumps(spec), capture_output=True, text=True, check=True,
    )
    return result.stdout.rstrip("\n") + "\n" + _values(spec)


def _key(name: str) -> str:
    return name if re.fullmatch(r"[A-Za-z_$][A-Za-z0-9_$]*", name) else json.dumps(name)


def _ts(value, indent: int = 0) -> str:
    """``value`` (JSON data) as a TypeScript literal, a flat object or list on one line."""
    pad = "  " * indent
    nested = isinstance(value, (dict, list))
    if isinstance(value, dict):
        if not value:
            return "{}"
        if not any(isinstance(v, (dict, list)) for v in value.values()):
            return "{ " + ", ".join(f"{_key(k)}: {json.dumps(v)}" for k, v in value.items()) + " }"
        return "{\n" + "\n".join(f"{pad}  {_key(k)}: {_ts(v, indent + 1)}," for k, v in value.items()) + f"\n{pad}}}"
    if isinstance(value, list):
        if not value:
            return "[]"
        if not any(isinstance(v, (dict, list)) for v in value):
            return "[" + ", ".join(json.dumps(v) for v in value) + "]"
        return "[\n" + "\n".join(f"{pad}  {_ts(v, indent + 1)}," for v in value) + f"\n{pad}]"
    assert not nested
    return json.dumps(value)


def _values(spec: dict) -> str:
    lines = [
        "",
        "// ---- Values (cellmap_flow/schema.py): the constants both modes share, the route",
        "// table and the chain steps' parameters, beside the types above.",
        "",
    ]
    for name, value in spec["x-constants"].items():
        lines.append(f"/** {CONSTANT_DOCS[name]} */")
        if name == "FINETUNE_DEFAULTS":
            lines.append(f"export const {name} = {_ts(value)} as const;")
        elif isinstance(value, list):
            lines.append(f"export const {name}: readonly number[] = {_ts(value)};")
        else:
            lines.append(f"export const {name} = {_ts(value)};")
    lines += [
        "",
        'export type Availability = Route["availability"];',
        "/** Every route of the dashboard and where it is answered (schema.py AVAILABILITY). */",
        f"export const ROUTES: readonly Route[] = {_ts(spec['x-routes'])};",
        "",
        "/** The input normalizers and the postprocessors, with their parameters' defaults. */",
        f"export const INPUT_NORMALIZERS: readonly StepSpec[] = {_ts(spec['x-steps']['input_norm'])};",
        f"export const POSTPROCESSORS: readonly StepSpec[] = {_ts(spec['x-steps']['postprocess'])};",
        "",
    ]
    return "\n".join(lines)


# --- Markdown -----------------------------------------------------------------------


def markdown(spec: Optional[dict] = None) -> str:
    """The route table as a Markdown page (docs/api.md)."""
    spec = schema() if spec is None else spec
    counts = {kind: sum(r["availability"] == kind for r in spec["x-routes"]) for kind in ("both", "stub", "server", "browser")}
    lines = [
        "# The dashboard's routes",
        "",
        "Generated from `cellmap_flow/schema.py` by `npm run gen` in web/ (or "
        "`python -m cellmap_flow.schema --docs docs/api.md`); do not edit. The method and "
        "path are the Flask app's; *answered by* says which mode answers the route: "
        f"**both** ({counts['both']}) the server dashboard and the browser app, **stub** ({counts['stub']}) "
        "both, the browser with a fixed or empty answer in the server's shape, "
        f"**server** ({counts['server']}) the server dashboard only (the browser answers "
        f"`{{success: false, available: false, error}}`), **browser** ({counts['browser']}) the browser app "
        "only. *Body* names the pydantic model of `cellmap_flow/dashboard/requests.py` (or the route's "
        "file) the body is parsed with; the generated `web/src/generated/cellmap-flow.ts` has its type.",
        "",
        "| Method | Path | Answered by | Body |",
        "|---|---|---|---|",
    ]
    for route in spec["x-routes"]:
        lines.append(f"| {route['method']} | `{route['path']}` | {route['availability']} | "
                     f"{'`' + route['request'] + '`' if route['request'] else ''} |")
    return "\n".join(lines) + "\n"


# --- The command --------------------------------------------------------------------


def outputs(out_dir: Optional[Path], docs: Optional[Path]) -> dict[Path, str]:
    """{path: content} of what --out and --docs write."""
    spec = schema()
    files = {}
    if out_dir:
        files[out_dir / SCHEMA_NAME] = dumps(spec)
        files[out_dir / TS_NAME] = typescript(spec)
    if docs:
        files[docs] = markdown(spec)
    return files


def check(files: dict[Path, str]) -> list[str]:
    """A unified diff per file of ``files`` whose content on disk differs."""
    diffs = []
    for path, content in files.items():
        current = path.read_text() if path.exists() else ""
        if current != content:
            diffs.append("".join(difflib.unified_diff(
                current.splitlines(True), content.splitlines(True), f"{path} (committed)", f"{path} (generated)")))
    return diffs


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m cellmap_flow.schema",
        description="The JSON Schema of the dashboard's request bodies, routes and shared constants, "
                    "and the TypeScript generated from it. With no option, print the schema.",
    )
    parser.add_argument("--out", type=Path, help=f"write {SCHEMA_NAME} and {TS_NAME} into this directory (web/src/generated)")
    parser.add_argument("--docs", type=Path, help="write the route table as Markdown to this file (docs/api.md)")
    parser.add_argument("--check", action="store_true",
                        help="instead of writing, say whether the committed files are current: exit 1 with a diff when not")
    args = parser.parse_args(argv)
    if args.check:
        diffs = check(outputs(args.out or GENERATED_DIR, args.docs or DOCS_FILE))
        for diff in diffs:
            sys.stdout.write(diff)
        if diffs:
            print(f"\n{len(diffs)} generated file(s) are stale; regenerate with {REGENERATE}")
            return 1
        print("the generated files are current")
        return 0
    if not args.out and not args.docs:
        sys.stdout.write(dumps())
        return 0
    for path, content in outputs(args.out, args.docs).items():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)
        print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
