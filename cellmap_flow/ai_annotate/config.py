"""The server-side config that turns AI-assisted annotation on, and what it allows.

The feature sends EM data to a hosted model, which costs money and moves data
off site, so it is off unless the user who runs the dashboard writes a config
file saying which providers and models may be used. The browser only ever
picks a provider id and model id from this file's lists; endpoints, projects
and credentials come from here and from the environment, never from a request.

The file holds no secrets. A key-based provider names an environment variable
(``api_key_env``) or a key file (``api_key_file``); an inline ``api_key`` is
a config error, so a key cannot end up in a file that gets copied, shared or
committed. Error messages name the setting that is wrong, never its value,
in case the value is a key pasted in the wrong place.

The file is read on each request that needs it (it is small), so an edit
takes effect without restarting the dashboard.
"""

import os
import posixpath
import re
from dataclasses import dataclass
from pathlib import Path

import yaml

from cellmap_flow.ai_annotate.errors import AIAnnotateError

CONFIG_ENV = "CELLMAP_FLOW_AI_ANNOTATE_CONFIG"
DOCS = "docs/ai_annotate.md"

DEFAULT_DAILY_CALL_LIMIT = 200
DEFAULT_CROP_SIZE_PX = 512
DEFAULT_VERTEX_LOCATION = "global"
DEFAULT_TIMEOUT_S = 120

_TOP_LEVEL_KEYS = {
    "enabled",
    "daily_call_limit",
    "crop_size_px",
    "allowed_dataset_prefixes",
    "default_provider",
    "providers",
}

# The options each provider type accepts besides ``type`` and ``models``. An
# unknown option is an error rather than ignored: a misspelt ``locaton`` would
# otherwise silently send data to the default location.
PROVIDER_OPTIONS = {
    "vertex_gemini": {"project", "location", "timeout_s", "credentials_file"},
    "fake": set(),
}

# Option names that would mean a secret written into the file. Refused with a
# message saying where a key belongs instead.
_SECRET_OPTION_NAMES = {"api_key", "key", "secret", "token", "password", "credentials"}

_ID_RE = re.compile(r"^[A-Za-z0-9_-]{1,64}$")
_MODEL_RE = re.compile(r"^[A-Za-z0-9._@/-]{1,200}$")
_LOCATION_RE = re.compile(r"^[a-z0-9-]{1,64}$")
_PROJECT_RE = re.compile(r"^[A-Za-z0-9._:-]{1,128}$")


def config_path():
    """Where the config is read from: ``$CELLMAP_FLOW_AI_ANNOTATE_CONFIG``, else
    ``~/.cellmap_flow/ai_annotate.yaml``."""
    override = os.environ.get(CONFIG_ENV)
    if override:
        return Path(override).expanduser()
    return Path.home() / ".cellmap_flow" / "ai_annotate.yaml"


@dataclass(frozen=True)
class ProviderConfig:
    """One provider the config allows: its id, type, models, and the rest of
    its settings (``options``: project, location, timeout_s, ...)."""

    id: str
    type: str
    models: tuple
    options: dict

    def destination(self):
        """Where a request to this provider sends the data, in words, for the
        user to acknowledge before the first call. Names the location but not
        the project: the user needs to know which company and region, not an
        account id."""
        if self.type == "vertex_gemini":
            location = self.options.get("location", DEFAULT_VERTEX_LOCATION)
            if location == "global":
                return "Google Cloud Vertex AI, location global (Google may process the data in any region)"
            return f"Google Cloud Vertex AI, location {location}"
        if self.type == "fake":
            return "nowhere (fake provider, runs locally)"
        return f"provider type {self.type}"


@dataclass(frozen=True)
class AIAnnotateConfig:
    """The parsed config file. Built only by ``load_config``, which validates it."""

    enabled: bool
    providers: dict
    default_provider: str
    daily_call_limit: int
    crop_size_px: int
    allowed_dataset_prefixes: tuple

    def provider(self, provider_id):
        """The provider with this id. Raises a config error (400: the id came
        from the browser) when the config has no such provider."""
        try:
            return self.providers[provider_id]
        except (KeyError, TypeError):
            raise AIAnnotateError(
                "config", "That AI provider is not in the AI-annotate config.", http_status=400
            ) from None

    def check_model(self, provider_id, model):
        """Raise unless ``model`` is one of the provider's listed models."""
        if model not in self.provider(provider_id).models:
            raise AIAnnotateError(
                "config", "That model is not listed for this provider in the AI-annotate config.", http_status=400
            )

    def check_dataset(self, dataset_path):
        """Refuse a dataset outside ``allowed_dataset_prefixes``, when that list is set.

        Local paths are normalised first, so ``/allowed/../elsewhere`` does not
        pass as ``/allowed``. A URL is not rewritten (normpath would mangle
        it), so one with a ``.`` or ``..`` segment, plain or percent-encoded,
        is refused outright: an HTTP client resolves those, and
        ``https://host/allowed/../secret`` would read ``/secret``. A prefix
        matches as text: end it with ``/`` to mean one directory and not also
        its siblings that share the name's start.
        """
        if not self.allowed_dataset_prefixes:
            return
        path = _normalise(str(dataset_path))
        if "://" in path and _has_dot_segment(path):
            raise AIAnnotateError(
                "refused",
                "This dataset's URL has a '.' or '..' segment, so whether it is under the AI-annotate "
                "config's allowed_dataset_prefixes cannot be told; open it by its plain URL.",
            )
        if not any(path.startswith(_normalise(p)) for p in self.allowed_dataset_prefixes):
            raise AIAnnotateError(
                "refused",
                "This dataset is not under one of the AI-annotate config's allowed_dataset_prefixes, "
                "so it may not be sent to a hosted model.",
            )

    def public(self):
        """What the browser may see: provider ids, types, models and
        destinations, and the limits. No other option value (project,
        timeouts, key variable or file names) is included."""
        return {
            "providers": [
                {"id": p.id, "type": p.type, "models": list(p.models), "destination": p.destination()}
                for p in self.providers.values()
            ],
            "default_provider": self.default_provider,
            "daily_call_limit": self.daily_call_limit,
            "crop_size_px": self.crop_size_px,
        }


def _normalise(path):
    """``path`` with ``..`` and doubled slashes resolved, a trailing slash
    kept; URLs (``s3://...``) are left alone, as normpath would mangle them."""
    if "://" in path:
        return path
    normal = posixpath.normpath(path)
    if path.endswith("/") and not normal.endswith("/"):
        normal += "/"
    return normal


def _has_dot_segment(url):
    """Whether ``url``'s path has a ``.`` or ``..`` segment, plain or percent-encoded."""
    path = url.split("://", 1)[1].split("?", 1)[0].split("#", 1)[0]
    segments = re.split(r"/|%2f", path, flags=re.IGNORECASE)
    return any(re.fullmatch(r"(\.|%2e){1,2}", segment, flags=re.IGNORECASE) for segment in segments)


def _bad(message):
    return AIAnnotateError("config", f"AI-annotate config {config_path()}: {message}")


def load_config():
    """The config, or None when the file does not exist or says ``enabled: false``.

    Raises a config error for a file that exists but is malformed, naming the
    problem and never echoing a value from the file.
    """
    path = config_path()
    try:
        text = path.read_text()
    except FileNotFoundError:
        return None
    except OSError:
        raise _bad("the file could not be read") from None
    try:
        raw = yaml.safe_load(text)
    except yaml.YAMLError as e:
        # The YAML error's own text quotes the offending line, which could be
        # a key; say only where it is.
        mark = getattr(e, "problem_mark", None)
        where = f" (line {mark.line + 1})" if mark is not None else ""
        raise _bad(f"not valid YAML{where}") from None
    if raw is None:
        raw = {}
    if not isinstance(raw, dict):
        raise _bad("the top level must be a mapping of settings")
    unknown = sorted(str(k) for k in raw if k not in _TOP_LEVEL_KEYS)
    if unknown:
        raise _bad(f"unknown setting(s) {', '.join(unknown)}")
    enabled = raw.get("enabled", False)
    if not isinstance(enabled, bool):
        raise _bad("enabled must be true or false")
    if not enabled:
        # Off: the rest may be half-written, and nothing will use it.
        return None

    providers = _parse_providers(raw.get("providers"))
    daily_call_limit = _int_setting(raw, "daily_call_limit", DEFAULT_DAILY_CALL_LIMIT, 0, 100_000)
    crop_size_px = _int_setting(raw, "crop_size_px", DEFAULT_CROP_SIZE_PX, 16, 4096)
    prefixes = raw.get("allowed_dataset_prefixes") or []
    if not isinstance(prefixes, list) or not all(isinstance(p, str) and p for p in prefixes):
        raise _bad("allowed_dataset_prefixes must be a list of paths")
    default_provider = raw.get("default_provider")
    if default_provider is None:
        default_provider = next(iter(providers))
    elif default_provider not in providers:
        raise _bad("default_provider is not one of the providers")

    return AIAnnotateConfig(
        enabled=True,
        providers=providers,
        default_provider=default_provider,
        daily_call_limit=daily_call_limit,
        crop_size_px=crop_size_px,
        allowed_dataset_prefixes=tuple(prefixes),
    )


def _int_setting(raw, name, default, lowest, highest):
    value = raw.get(name, default)
    # bool is an int subclass; "true" is not a limit.
    if isinstance(value, bool) or not isinstance(value, int) or not lowest <= value <= highest:
        raise _bad(f"{name} must be a whole number from {lowest} to {highest}")
    return value


def _parse_providers(raw):
    if not isinstance(raw, dict) or not raw:
        raise _bad("providers must map at least one provider id to its settings")
    providers = {}
    for provider_id, settings in raw.items():
        if not isinstance(provider_id, str) or not _ID_RE.match(provider_id):
            raise _bad("provider ids may use only letters, digits, - and _ (at most 64)")
        providers[provider_id] = _parse_provider(provider_id, settings)
    return providers


def _parse_provider(provider_id, settings):
    where = f"providers.{provider_id}"
    if not isinstance(settings, dict):
        raise _bad(f"{where} must be a mapping of settings")
    secret_like = sorted(str(k) for k in settings if str(k).lower() in _SECRET_OPTION_NAMES)
    if secret_like:
        raise _bad(
            f"{where}.{secret_like[0]}: secrets may not be written in this file. Put an API key in an "
            "environment variable (api_key_env) or a file only you can read (api_key_file); "
            f"Vertex AI needs no key at all (see {DOCS})"
        )
    provider_type = settings.get("type")
    if provider_type not in PROVIDER_OPTIONS:
        raise _bad(f"{where}.type must be one of {', '.join(sorted(PROVIDER_OPTIONS))}")
    unknown = sorted(str(k) for k in settings if k not in {"type", "models"} | PROVIDER_OPTIONS[provider_type])
    if unknown:
        raise _bad(f"{where}: unknown setting(s) {', '.join(unknown)} for type {provider_type}")
    models = settings.get("models")
    if (
        not isinstance(models, list)
        or not models
        or not all(isinstance(m, str) and _MODEL_RE.match(m) for m in models)
    ):
        raise _bad(f"{where}.models must be a non-empty list of model names")
    options = {k: v for k, v in settings.items() if k not in {"type", "models"}}
    _check_options(where, provider_type, options)
    return ProviderConfig(id=provider_id, type=provider_type, models=tuple(models), options=options)


def _check_options(where, provider_type, options):
    """Type- and shape-check a provider's options. Values are checked against
    narrow patterns because they end up in requests the dashboard makes."""
    if provider_type != "vertex_gemini":
        return
    project = options.get("project")
    if project is not None and (not isinstance(project, str) or not _PROJECT_RE.match(project)):
        raise _bad(f"{where}.project must be a Google Cloud project id")
    location = options.setdefault("location", DEFAULT_VERTEX_LOCATION)
    if not isinstance(location, str) or not _LOCATION_RE.match(location):
        raise _bad(f"{where}.location must be a Google Cloud location such as global or us-central1")
    timeout_s = options.setdefault("timeout_s", DEFAULT_TIMEOUT_S)
    if isinstance(timeout_s, bool) or not isinstance(timeout_s, (int, float)) or not 1 <= timeout_s <= 600:
        raise _bad(f"{where}.timeout_s must be a number of seconds from 1 to 600")
    # The path of a credentials file, not the credentials: the file itself is
    # read, and its permissions checked, only when a call is made
    # (secrets.load_google_credentials). Left out, the SDK finds Application
    # Default Credentials by itself.
    credentials_file = options.get("credentials_file")
    if credentials_file is not None and (
        not isinstance(credentials_file, str) or not credentials_file.strip() or len(credentials_file) > 4096
    ):
        raise _bad(f"{where}.credentials_file must be the path of a Google credentials file")


def disabled_reason():
    """Why the feature is off, and how to turn it on, for the dashboard to show.

    Called when ``load_config`` returned None; a malformed file raises instead.
    """
    path = config_path()
    if not path.exists():
        return (
            f"AI-assisted annotation is off: there is no config file at {path}. Create one (or point "
            f"{CONFIG_ENV} at one) listing the providers and models to allow; see {DOCS}."
        )
    return f"AI-assisted annotation is off: {path} says enabled: false. Set enabled: true to turn it on; see {DOCS}."
