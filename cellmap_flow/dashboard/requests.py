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
- ``FinetuneSubmit``: POST /api/finetune/submit
- ``FinetuneRestart``: POST /api/finetune/job/<job_id>/restart
- ``CreateVolume``: POST /api/finetune/create-volume
- ``LoadCrops``: POST /api/finetune/load-crops
- ``BlockwiseValidate``, ``BlockwiseGenerate``, ``BlockwisePrecheck`` and
  ``BlockwiseSubmit``: POST /api/blockwise/{validate,generate,precheck,submit},
  which answer through ``check`` instead (see there)
"""

from typing import Annotated, Any, Optional

from flask import jsonify
from pydantic import (
    BaseModel,
    BeforeValidator,
    ConfigDict,
    Field,
    ValidationError,
    ValidationInfo,
    field_validator,
    model_validator,
)

from cellmap_flow.jobs.settings import SERVER_CONFIG_KEYS
from cellmap_flow.jobs.site import current_site


def _whole_number(value, info: ValidationInfo):
    # int() as the routes always parsed the counts: "12" and 12.0 are 12.
    try:
        return int(value)
    except (TypeError, ValueError):
        raise ValueError(f"{info.field_name} must be a whole number, got {value!r}")


WholeNumber = Annotated[int, BeforeValidator(_whole_number)]


def _required(name, strip=False, message=None):
    """A validator refusing an empty ``name`` (``message``, by default "<name>
    is required"); with ``strip``, surrounding blanks are taken off a string
    first."""
    def check(value):
        if strip and isinstance(value, str):
            value = value.strip()
        if not value:
            raise ValueError(message or f"{name} is required")
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
    """Any of the saved settings (jobs.settings.SERVER_CONFIG_DEFAULTS); only those
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


# --- The finetune tab's ------------------------------------------------------------


def _form_number(kind, default):
    """A number field of the finetune tab's forms, of type ``kind`` (int or float).

    The form sends what its fields hold, so an emptied field arrives as null
    or "", and is ``default`` here (it reached the trainer as "--num-epochs
    None", which failed only once the job ran). A whole number may come as
    "25" or 25.0, but not 2.5. Anything else is refused with the field's
    name and what was sent.
    """
    what = "a whole number" if kind is int else "a number"

    def read(value, info: ValidationInfo):
        if value is None or value == "":
            return default
        try:
            number = float(value)
        except (TypeError, ValueError):
            number = None
        if number is None or (kind is int and not number.is_integer()):
            raise ValueError(f"{info.field_name} must be {what}, got {value!r}")
        return int(number) if kind is int else number

    return Annotated[kind if default is not None else Optional[kind], Field(default=default), BeforeValidator(read)]


_OVERRIDES = ("patches_per_epoch", "rehearsal_fraction")


class _ManifestOverrides(BaseModel):
    """What a run may change in its session's manifest, at submit and at each
    restart: ``patches_per_epoch`` (a count, or 0 for one patch per
    populated chunk, stored as None) and ``rehearsal_fraction`` (between 0
    and 1; 0 turns rehearsal off without dropping the good regions). One
    that is absent or blank leaves the manifest's value as it is; see
    ``overrides()``. Every other key of the body is kept as sent."""

    model_config = ConfigDict(extra="allow")

    patches_per_epoch: Optional[int] = None
    rehearsal_fraction: Optional[float] = None

    @model_validator(mode="before")
    @classmethod
    def _blank_is_not_given(cls, body):
        if isinstance(body, dict):
            body = {k: v for k, v in body.items() if not (k in _OVERRIDES and v in (None, ""))}
        return body

    @field_validator("patches_per_epoch", mode="before")
    @classmethod
    def _count_or_auto(cls, value):
        try:
            count = int(value)
        except (TypeError, ValueError):
            count = -1
        if count < 0:
            raise ValueError("patches_per_epoch must be a non-negative integer")
        return None if count == 0 else count

    @field_validator("rehearsal_fraction", mode="before")
    @classmethod
    def _fraction(cls, value):
        try:
            fraction = float(value)
        except (TypeError, ValueError):
            fraction = -1.0
        if not 0.0 <= fraction <= 1.0:
            raise ValueError("rehearsal_fraction must be a number between 0 and 1")
        return fraction

    def overrides(self) -> dict:
        """{override: value} for each one the body gave."""
        return {key: getattr(self, key) for key in _OVERRIDES if key in self.model_fields_set}


class FinetuneSubmit(_ManifestOverrides):
    """A new training job: ``model_name`` trained on the corrections at
    ``corrections_path``, a session's corrections directory or the base
    output path of its sessions.

    The defaults are the form's. The rest of the fields are passed on as
    sent, and the route adjusts the loss for painted annotations and for a
    distance target.
    """

    model_name: Annotated[str, _required("model_name")] = Field(None, validate_default=True)
    corrections_path: Annotated[str, _required(
        "corrections_path",
        message="corrections_path is required. Please specify the output path where annotation crops are saved.",
    )] = Field(None, validate_default=True)
    lora_r: _form_number(int, 8)
    num_epochs: _form_number(int, 10)
    batch_size: _form_number(int, 2)
    learning_rate: _form_number(float, 1e-4)
    loss_type: Any = "mse"
    label_smoothing: _form_number(float, 0.1)
    # None leaves the weight to the trainer; 0 switches it off.
    distillation_lambda: _form_number(float, None)
    distillation_scope: Any = "unlabeled"
    margin: _form_number(float, 0.3)
    balance_classes: Any = False
    # Default off: these interactive runs are a few dozen gradient steps,
    # where augmentation adds variance without the many repeat views it
    # needs to pay for itself.
    augment: Any = False
    output_type: Any = None
    offsets: Any = None
    select_channel: _form_number(int, None)
    checkpoint_path: Any = None
    auto_serve: Any = True
    queue: Any = Field(default_factory=lambda: current_site().default_queue)
    charge_group: Any = None


class FinetuneRestart(_ManifestOverrides):
    """The next iteration of a job: the manifest overrides, and the training
    settings to change (common.build_restart_params picks those; the
    trainer checks them)."""


class CreateVolume(BaseModel):
    """A new, empty annotation volume for ``model_name``, in a new session
    under ``output_path`` (by default ~/.cellmap_flow/corrections)."""

    model_name: Annotated[str, _required("model_name")] = Field(None, validate_default=True)
    output_path: Any = None


class LoadCrops(BaseModel):
    """Crops imported into the session's annotation volume for ``model_name``:
    ``yaml`` is the YAML's text or its path, ``output_path`` as for
    CreateVolume, and ``load_id`` names the import's progress."""

    yaml: Annotated[str, _required("yaml", message="Missing 'yaml' field")] = Field(None, validate_default=True)
    model_name: Annotated[str, _required("model_name", message="Missing 'model_name' field")] = Field(
        None, validate_default=True)
    output_path: Any = None
    load_id: Any = None


# --- The pipeline builder's blockwise steps -----------------------------------------
#
# The blockwise routes answer a body they cannot take with a 200 whose flag,
# "valid" from validate and "success" from the others, is false beside the
# error: the builder reads the flag at each step. So they use check(), which
# gives parse()'s message without its 400.


def check(model, body):
    """``(model instance, None)`` for a valid ``body``; else ``(None, what is
    wrong)``, the message parse() would answer with."""
    if not isinstance(body, dict):
        return None, "expected a JSON object"
    try:
        return model.model_validate(body), None
    except ValidationError as e:
        return None, _message(e)


class PipelineNode(BaseModel):
    """A node of the builder's pipeline (state.js), ``{id, name, params,
    position}``, of which the routes read the name and the params."""

    name: Any = None
    params: dict = {}


class ChainNode(PipelineNode):
    """A normalizer or postprocessor node, whose params may be null for none."""

    params: Optional[dict] = None


class ModelNode(PipelineNode):
    """A model node. It may carry the config the model was defined with (its
    ModelConfig.to_dict()), which stands in for params when it has none."""

    config: dict = {}

    def settings(self) -> dict:
        """Its params, or its config when it was sent without params."""
        return self.params if "params" in self.model_fields_set else self.config


class BlockwiseTaskSettings(BaseModel):
    """A blockwise-config node's params: what a task and its master are
    submitted with. Each is required, and taken as sent (precheck checks
    the task's)."""

    charge_group: Any
    queue: Any
    nb_workers: Any
    nb_cores_worker: Any
    nb_cores_master: Any
    tmp_dir: Any


class BlockwiseConfigNode(PipelineNode):
    params: BlockwiseTaskSettings


class BlockwisePipeline(BaseModel):
    """The builder's pipeline, one list of nodes per node type, as the
    blockwise routes read it: the first input, output and blockwise-config
    node are the ones used, and the models are run on the chain the
    normalizers and postprocessors make. The four lists must have a node, and
    the input and output a dataset_path, checked in that order."""

    inputs: Annotated[list[PipelineNode], _required("inputs", message="No input nodes defined")] = Field(
        None, validate_default=True)
    outputs: Annotated[list[PipelineNode], _required("outputs", message="No output nodes defined")] = Field(
        None, validate_default=True)
    models: Annotated[list[ModelNode], _required("models", message="No models defined")] = Field(
        None, validate_default=True)
    blockwise_config: Annotated[list[BlockwiseConfigNode], _required(
        "blockwise_config", message="No blockwise configuration defined")] = Field(None, validate_default=True)
    normalizers: list[ChainNode] = []
    postprocessors: list[ChainNode] = []
    # How the models' outputs are merged, when there are several.
    model_mode: Any = ""

    @model_validator(mode="after")
    def _dataset_paths(self):
        if not self.inputs[0].params.get("dataset_path"):
            raise ValueError("Input node missing dataset_path")
        if not self.outputs[0].params.get("dataset_path"):
            raise ValueError("Output node missing dataset_path")
        return self


class BlockwiseValidate(BaseModel):
    """Whether ``pipeline`` is ready to run blockwise."""

    pipeline: BlockwisePipeline = Field({}, validate_default=True)


class BlockwiseGenerate(BlockwiseValidate):
    """The task YAML(s) for ``pipeline``; ``job_name`` names the task."""

    job_name: Any = ""


class BlockwisePrecheck(BaseModel):
    """The task YAMLs generate wrote, to check."""

    yaml_paths: Annotated[list[str], _required(
        "yaml_paths", message="No YAML paths provided. Please generate task first.")] = Field(
        None, validate_default=True)


class BlockwiseSubmit(BlockwiseGenerate):
    """The task to submit: the YAMLs precheck passed and the name generate gave
    them, or else, when none are given or they are not all there,
    ``pipeline``'s, generated anew. ``yaml_paths`` is a list of paths, as for
    precheck."""

    yaml_paths: Optional[list[str]] = None
    task_name: Any = None
