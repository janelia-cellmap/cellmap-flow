#!/usr/bin/env python
"""Render the dashboard's index.html for the browser build.

The browser mode serves the same dashboard as the Flask app: the same Jinja
templates (cellmap_flow/dashboard/templates), the same JavaScript and CSS
(cellmap_flow/dashboard/static). The Flask app renders index.html on every
request with the session's state in it; here it is rendered once, at build
time, with the state a fresh browser session has (no dataset, no models, the
default chains), and ``mode="browser"`` so the page can hide what only a
server can do (the templates mark those controls ``server-only``, and
web/src/browser.css hides them under ``data-mode="browser"``).

The rendered page names ``app_entry`` (web/src/main.ts) as its script instead
of the dashboard's own main.js: the entry starts the in-page backend and then
loads that same main.js, so the dashboard's first requests already have
someone to answer them.

Usage (from web/)::

    python build/render.py                 # writes web/index.html
    python build/render.py --out -         # writes the page to stdout
    python build/render.py --check         # exits 1 if web/index.html is stale

``npm run render`` runs build/export_data.py after it, so the data the app
shares with Python (web/src/generated/python-data.json) is brought up to
date with the page.

Renders the cellmap_flow beside this script (the repository root, two
directories up, put first on sys.path, so a worktree renders its own templates
and not an installed copy's); its dependencies must be importable (the pixi
``default`` environment has them).
tests/web/test_render.py renders the page in-process and checks it.
"""
import argparse
import json
import sys
from pathlib import Path

from jinja2 import Environment, FileSystemLoader, StrictUndefined, select_autoescape
from markupsafe import Markup

# The cellmap_flow this script renders is the one it ships with: the package at
# the repository root, ahead of any installed copy, so a worktree (or a checkout
# beside a pip-installed cellmap_flow) renders its own templates.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

HERE = Path(__file__).resolve().parent
WEB = HERE.parent

# The exporter beside this script. A script run finds it on sys.path already;
# the tests that import this file by path find it through this.
sys.path.insert(1, str(HERE))
import export_data  # noqa: E402

# The chains a fresh browser session starts with, ticked on the Input and
# Postprocess tabs with these values (the app's session starts with the same).
DEFAULT_CHAIN = export_data.DEFAULT_CHAIN

# The page's module script and where the dashboard's static files are served,
# relative to the page, as vite.config.ts and build/vite-plugins.ts arrange it.
APP_ENTRY = "./src/main.ts"
STATIC_PREFIX = "./static/"


def tojson(value):
    """Flask's ``tojson`` filter: JSON that is safe inside a <script> element.

    ``<``, ``>``, ``&`` and ``'`` become unicode escapes, so a value can never
    close the script early, and the result is marked safe so autoescaping
    leaves the quotes alone.
    """
    text = json.dumps(value)
    for char, escaped in (("<", "\\u003c"), (">", "\\u003e"), ("&", "\\u0026"), ("'", "\\u0027")):
        text = text.replace(char, escaped)
    return Markup(text)


def make_url_for(static_prefix):
    """Flask's ``url_for``, for the one endpoint the templates use: static files,
    which the browser build serves beside the page under ``static_prefix``."""

    def url_for(endpoint, **values):
        if endpoint != "static":
            raise ValueError(f"the browser build cannot resolve url_for({endpoint!r})")
        return static_prefix + values["filename"]

    return url_for


def template_dir():
    """The dashboard's templates, from the cellmap_flow that is importable."""
    import cellmap_flow

    return Path(cellmap_flow.__file__).resolve().parent / "dashboard" / "templates"


def browser_context(app_entry=APP_ENTRY, static_prefix=STATIC_PREFIX, chain=DEFAULT_CHAIN):
    """The template variables index.html and its partials need, as a fresh
    browser session has them.

    The names are those routes/index_page.py:index passes to render_template,
    plus ``mode`` and ``app_entry``, which Flask leaves undefined. Keep the two
    in step: the template is rendered with StrictUndefined, so a variable that
    the templates come to need and this context lacks fails the render (and
    tests/web/test_render.py) with its name, rather than rendering blank.
    """
    from cellmap_flow.dashboard.routes.index_page import chain_items
    from cellmap_flow.norm.input_normalize import get_input_normalizers
    from cellmap_flow.pipeline_spec import PipelineSpec
    from cellmap_flow.post.postprocessors import get_postprocessors_list

    # chain_items takes live steps (it calls their to_dict), as the Flask
    # route hands it the session's: build the default chain's the same way, so
    # a parameter is shown exactly as the dashboard would show it.
    input_norms, postprocess = PipelineSpec(chain["input_norm"], chain["postprocess"]).build()

    def in_browser(ops):
        # Only the ops the browser runs (their ``browser`` flag): one it
        # cannot run would be refused when ticked.
        return [op for op in ops if op["browser"]]

    return dict(
        mode="browser",
        app_entry=app_entry,
        url_for=make_url_for(static_prefix),
        # No viewer yet: the page shows the connect panel, and the in-page
        # backend puts the viewer in when a dataset is opened.
        neuroglancer_url="",
        # The default chain ticked, with its values; every other op the
        # browser runs after it, unticked.
        input_norm_items=chain_items(in_browser(get_input_normalizers()), input_norms),
        postprocess_items=chain_items(in_browser(get_postprocessors_list()), postprocess),
        # The Local Catalog lists paths on the cluster's file systems, which a
        # browser cannot read; the browser's models come from Hugging Face.
        model_catalog={},
        default_models=[],
        default_hf_repos=[],
        default_bioimage_models=[],
        cellpose_panel={"models": [], "running": []},
        resample=False,
        # LSF settings mean nothing in a browser.
        ask_server_config=False,
        on_cluster=False,
    )


def render(template_dir, context):
    """index.html rendered with ``context``, as Flask would render it but with
    this file's ``tojson`` and the context's own ``url_for``."""
    env = Environment(
        loader=FileSystemLoader(str(template_dir)),
        undefined=StrictUndefined,
        autoescape=select_autoescape(["html"]),
    )
    env.filters["tojson"] = tojson
    return env.get_template("index.html").render(**context)


def render_browser_page(app_entry=APP_ENTRY, static_prefix=STATIC_PREFIX):
    """The browser build's page: the dashboard's index.html with the browser context."""
    return render(template_dir(), browser_context(app_entry, static_prefix))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", default=str(WEB / "index.html"),
                        help="where to write the page, or - for stdout (default: web/index.html)")
    parser.add_argument("--check", action="store_true",
                        help="write nothing; exit 1 if --out does not hold what would be rendered")
    parser.add_argument("--app-entry", default=APP_ENTRY, help="the page's module script")
    parser.add_argument("--static-prefix", default=STATIC_PREFIX,
                        help="where the dashboard's static files are served, relative to the page")
    args = parser.parse_args(argv)

    html = render_browser_page(args.app_entry, args.static_prefix)
    if args.out == "-":
        sys.stdout.write(html)
        return 0
    out = Path(args.out)
    if args.check:
        current = out.read_text() if out.exists() else None
        if current == html:
            print(f"{out} is up to date")
            return 0
        print(f"{out} is {'missing' if current is None else 'stale'}: run build/render.py", file=sys.stderr)
        return 1
    out.write_text(html)
    print(f"rendered {template_dir() / 'index.html'} -> {out} ({len(html):,} chars)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
