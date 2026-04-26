"""Render cellmap-flow's dashboard templates to a static HTML file.

Reads templates from `cellmap_flow/dashboard/templates/` (the live source of
truth) and writes the rendered output to `browser/public/dashboard.html`.
The static assets (CSS/JS/images) are exposed at `/dashboard-static/` via a
Vite plugin (see `vite.config.ts`).

Re-run this script whenever cellmap-flow's dashboard templates change. It's
also invoked by `npm run build` via the `prebuild` hook.

Usage:
    python browser/scripts/render-dashboard.py
"""

from __future__ import annotations

import sys
from pathlib import Path

from jinja2 import ChainableUndefined, Environment, FileSystemLoader, select_autoescape

# Try to import cellmap_flow so we can populate the same context the live
# Flask dashboard uses. Fall back to an empty/stub context if the package
# can't load (so the build still works without the cluster-side deps).
try:
    from cellmap_flow.norm.input_normalize import get_input_normalizers
    from cellmap_flow.post.postprocessors import get_postprocessors_list
    from cellmap_flow.models.model_merger import get_model_mergers_list

    def populate_context() -> dict:
        return {
            "neuroglancer_url": None,
            "inference_servers": None,
            "input_normalizers": get_input_normalizers(),
            "output_postprocessors": get_postprocessors_list(),
            "model_mergers": get_model_mergers_list(),
            "default_post_process": {},
            "default_input_norm": {},
            "model_catalog": {},
            "default_models": [],
            "default_hf_repos": [],
            "server_config_cached": True,
            # Pipeline-builder template variables — empty defaults so the
            # rendered HTML is well-formed; client-side JS populates these.
            "available_models": [],
            "current_edges": [],
            "current_inputs": [],
            "current_models": [],
            "current_normalizers": [],
            "current_outputs": [],
            "current_postprocessors": [],
            "dataset_path": "",
        }
except Exception as e:  # noqa: BLE001
    print(f"warning: could not import cellmap_flow ({e}); using empty context", file=sys.stderr)

    def populate_context() -> dict:
        return {
            "neuroglancer_url": None,
            "inference_servers": None,
            "input_normalizers": [],
            "output_postprocessors": [],
            "model_mergers": [],
            "default_post_process": {},
            "default_input_norm": {},
            "model_catalog": {},
            "default_models": [],
            "default_hf_repos": [],
            "server_config_cached": True,
            # Pipeline-builder template variables — empty defaults so the
            # rendered HTML is well-formed; client-side JS populates these.
            "available_models": [],
            "current_edges": [],
            "current_inputs": [],
            "current_models": [],
            "current_normalizers": [],
            "current_outputs": [],
            "current_postprocessors": [],
            "dataset_path": "",
        }


HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
TEMPLATES = REPO / "cellmap_flow" / "dashboard" / "templates"
# Render to the browser/ root so Vite picks them up as multi-page entries.
# The script tags then resolve through Vite's pipeline (otherwise public/
# files are served as-is and module imports break).
OUT_DIR = HERE.parent


def url_for(_endpoint: str, filename: str = "", **_kwargs: object) -> str:
    """Stand-in for Flask's url_for that resolves static asset paths to
    `/dashboard-static/<filename>` (matching the Vite middleware below).
    Anything that's not a static asset returns "#" so the rendered HTML is
    still well-formed."""
    if _endpoint == "static":
        return f"/dashboard-static/{filename}"
    return "#"


def render(template_name: str, out_filename: str) -> None:
    env = Environment(
        loader=FileSystemLoader(str(TEMPLATES)),
        autoescape=select_autoescape(["html", "xml"]),
        undefined=ChainableUndefined,
    )
    env.globals["url_for"] = url_for
    template = env.get_template(template_name)
    html = template.render(**populate_context())

    # Inject our browser-side shim that wires the dashboard's existing JS to
    # our in-browser /vz/ pipeline. Placed just before </body> so it runs
    # after the dashboard's inline scripts have bound their handlers.
    shim = '<script type="module" src="/src/dashboard-shim.ts"></script>\n'
    html = html.replace("</body>", f"  {shim}</body>", 1)

    out_path = OUT_DIR / out_filename
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(html)
    print(f"  {template_name} -> {out_path} ({len(html) / 1024:.1f} KiB)")


def main() -> None:
    if not TEMPLATES.exists():
        sys.exit(f"templates not found at {TEMPLATES}")
    print(f"rendering dashboard from {TEMPLATES}")
    render("index.html", "dashboard.html")
    render("pipeline_builder_v2.html", "pipeline_builder.html")
    print("done.")


if __name__ == "__main__":
    main()
