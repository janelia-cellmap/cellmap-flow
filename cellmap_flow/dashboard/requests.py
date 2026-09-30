"""The typed bodies of the dashboard's POST routes, and how a bad one is answered.

``parse(Model, body)`` gives ``(request, None)``, or ``(None, response)``
where the response is the 400 every route answers a bad body with,
``{"success": False, "error": <message>}``, before anything is changed. The
message names what is wrong: a body that is not a JSON object is "expected a
JSON object", a count that is not a whole number says so and shows what was
sent (the settings forms show it as it stands), and any other field's
message is pydantic's, after the field's name.

- ``ServerConfigUpdate``: POST /api/server-config
- ``BlockwiseSettings``: POST /api/blockwise-config
- ``CreateModelConfig``: POST /api/create-model-config
- ``SetData``: POST /api/set-data
"""

from typing import Annotated, Any

from flask import jsonify
from pydantic import BaseModel, BeforeValidator, ConfigDict, Field, ValidationError, ValidationInfo

from cellmap_flow.globals import SERVER_CONFIG_KEYS


def _whole_number(value, info: ValidationInfo):
    # int() as the routes always parsed the counts: "12" and 12.0 are 12.
    try:
        return int(value)
    except (TypeError, ValueError):
        raise ValueError(f"{info.field_name} must be a whole number, got {value!r}")


WholeNumber = Annotated[int, BeforeValidator(_whole_number)]


def _required(name, strip=False):
    """A validator refusing an empty ``name`` ("<name> is required"); with
    ``strip``, surrounding blanks are taken off a string first."""
    def check(value):
        if strip and isinstance(value, str):
            value = value.strip()
        if not value:
            raise ValueError(f"{name} is required")
        return value
    return BeforeValidator(check)


def _message(error: ValidationError) -> str:
    first = error.errors()[0]
    if "error" in first.get("ctx", {}):  # one of the validators above
        return str(first["ctx"]["error"])
    where = ".".join(str(part) for part in first["loc"])
    return f"{where}: {first['msg']}" if where else first["msg"]


def parse(model, body):
    """``(model instance, None)`` for a valid ``body``; else ``(None, the 400 response)``."""
    if not isinstance(body, dict):
        message = "expected a JSON object"
    else:
        try:
            return model.model_validate(body), None
        except ValidationError as e:
            message = _message(e)
    return None, (jsonify({"success": False, "error": message}), 400)


class ServerConfigUpdate(BaseModel):
    """Any of the saved settings (globals.SERVER_CONFIG_DEFAULTS); only those
    sent change. The counts must be whole numbers; the rest are kept as sent,
    and keys that are not settings are ignored."""

    model_config = ConfigDict(extra="allow")

    nb_cores_master: WholeNumber = None
    nb_cores_worker: WholeNumber = None
    nb_workers: WholeNumber = None

    def updates(self) -> dict:
        """{setting: value} for each setting the request sent."""
        sent = self.model_dump(exclude_unset=True)
        return {key: sent[key] for key in SERVER_CONFIG_KEYS if key in sent}


class BlockwiseSettings(BaseModel):
    """The pipeline builder's blockwise settings. All three counts are
    required, as whole numbers; a missing one is reported as None."""

    queue: Any = None
    charge_group: Any = None
    nb_cores_master: WholeNumber = Field(None, validate_default=True)
    nb_cores_worker: WholeNumber = Field(None, validate_default=True)
    nb_workers: WholeNumber = Field(None, validate_default=True)
    tmp_dir: Any = None
    blockwise_tasks_dir: Any = None


class CreateModelConfig(BaseModel):
    """A ModelConfig subclass's name and its constructor's parameters, as the
    builder's model form sends them (registry.coerce_form_params parses them)."""

    class_name: Annotated[str, _required("class_name")] = Field(None, validate_default=True)
    params: dict = {}


class SetData(BaseModel):
    """The dataset to open a new viewer on."""

    dataset_path: Annotated[str, _required("dataset_path", strip=True)] = Field(None, validate_default=True)
