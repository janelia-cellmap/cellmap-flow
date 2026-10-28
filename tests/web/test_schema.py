"""The contract both backends are held to: cellmap_flow/schema.py and what it generates.

The pydantic request models, the route table and the shared constants are
the spec. web/src/generated/cellmap-flow.{schema.json,ts} and docs/api.md
are generated from them and must be current; every route of the Flask app
is declared with where it is answered; the browser registers exactly the
routes declared for it; and the Finetune tab's form shows the defaults
FinetuneSubmit reads a blank field as.
"""

import re
import shutil
from html.parser import HTMLParser
from pathlib import Path

import pytest

from cellmap_flow import schema

ROOT = Path(__file__).resolve().parents[2]
ROUTES_TS = ROOT / "web" / "src" / "backend" / "routes"
STALE = f"is stale: regenerate with {schema.REGENERATE} and commit the result"


def test_the_committed_schema_is_current():
    assert (schema.GENERATED_DIR / schema.SCHEMA_NAME).read_text() == schema.dumps(), f"{schema.SCHEMA_NAME} {STALE}"


def test_the_route_docs_are_current():
    assert schema.DOCS_FILE.read_text() == schema.markdown(), f"docs/api.md {STALE}"


def test_the_generated_typescript_is_current():
    if not shutil.which("node") or not schema.JSON2TS.exists():
        pytest.skip("needs node and `npm ci` in web/ (CI's web job runs this check)")
    assert (schema.GENERATED_DIR / schema.TS_NAME).read_text() == schema.typescript(), f"{schema.TS_NAME} {STALE}"


def test_every_route_of_the_dashboard_is_declared():
    rules = schema.flask_rules()
    assert set(rules) <= {(route.method, route.path) for route in schema.routes(rules)}
    # A route added to the app without a declaration, and a declaration the
    # app has lost, are refused by name.
    with pytest.raises(LookupError, match=re.escape("POST /api/new-thing")):
        schema.routes([*rules, ("POST", "/api/new-thing")])
    with pytest.raises(LookupError, match=re.escape(f"{rules[0][0]} {rules[0][1]}")):
        schema.routes(rules[1:])


def _erase_params(path: str) -> str:
    """Flask's `/job/<job_id>/status` and the router's `/job/:id/status` as one."""
    return re.sub(r"<[^>]*>|:[A-Za-z_]\w*", "<>", path)


def _browser_routes() -> set[tuple[str, str]]:
    """(METHOD, path) of every route web/src/backend/routes registers on the router."""
    found = set()
    for file in sorted(ROUTES_TS.glob("*.ts")):
        text = file.read_text()
        base = re.search(r'^const BASE = "([^"]+)";', text, re.M)
        for method, literal, templated in re.findall(
                r'router\.(get|post|put|delete)\(\s*(?:"([^"]+)"|`\$\{BASE\}([^`]*)`)', text):
            assert literal or base, f"{file.name} registers `${{BASE}}...` without a BASE"
            found.add((method.upper(), _erase_params(literal or base.group(1) + templated)))
    return found


def test_the_browser_registers_the_routes_declared_for_it():
    declared = {(r.method, _erase_params(r.path)) for r in schema.routes() if r.availability != "server"}
    registered = _browser_routes()
    assert registered - declared == set(), \
        "routes web/src/backend/routes registers that schema.py's AVAILABILITY does not give the browser (both, stub or browser)"
    assert declared - registered == set(), \
        "routes AVAILABILITY gives the browser (both, stub or browser) that web/src/backend/routes does not register"


class _Form(HTMLParser):
    """Each input's value by id (a checkbox's: whether it is checked), and each select's selected option."""

    def __init__(self):
        super().__init__()
        self.values = {}
        self._select = None

    def handle_starttag(self, tag, attrs):
        a = dict(attrs)
        if tag == "input" and "id" in a:
            self.values[a["id"]] = ("checked" in a) if a.get("type") == "checkbox" else a.get("value")
        elif tag == "select":
            self._select = a.get("id")
        elif tag == "option" and self._select and "selected" in a:
            self.values[self._select] = a.get("value")

    def handle_endtag(self, tag):
        if tag == "select":
            self._select = None


# The Finetune tab's fields, by element id, and the FinetuneSubmit field each is sent as.
FORM_FIELDS = {
    "loraRank": "lora_r", "numEpochs": "num_epochs", "batchSize": "batch_size", "learningRate": "learning_rate",
    "learningRateEase": "learning_rate_schedule", "lossType": "loss_type", "marginValue": "margin", "labelSmoothing": "label_smoothing",
    "distillationLambda": "distillation_lambda", "distillationScope": "distillation_scope",
    "balanceClasses": "balance_classes", "augment": "augment",
}


def test_the_server_form_shows_what_a_blank_field_trains_with():
    """FinetuneSubmit's defaults had drifted from the form's (10 epochs, a batch
    of 2 and MSE against the form's 20, 8 and margin), so a field sent blank
    trained with values the form never showed."""
    from cellmap_flow.dashboard.app import app
    from cellmap_flow.dashboard.requests import FinetuneSubmit

    form = _Form()
    form.feed(app.jinja_env.get_template("_finetune_tab.html").render())  # no mode: the server's page
    defaults = {field: FinetuneSubmit.model_fields[field].default for field in FORM_FIELDS.values()}
    shown = {field: form.values[element] for element, field in FORM_FIELDS.items()}
    # Ease down to a tenth, ticked, sends the cosine schedule; unticked, constant.
    shown["learning_rate_schedule"] = "cosine" if shown["learning_rate_schedule"] else "constant"
    as_sent = {field: value if isinstance(value, bool) else type(defaults[field])(value) for field, value in shown.items()}
    assert as_sent == defaults
    assert set(FORM_FIELDS.values()) == set(schema.FINETUNE_FORM_FIELDS), "the fields exported to the browser"
