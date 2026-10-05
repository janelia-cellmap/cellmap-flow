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
- ``SubmitModels``: POST /api/models
- ``Equivalences``: POST /update/equivalences
- ``BbxGenerator``: POST /api/bbx-generator
- ``FinetuneSubmit``: POST /api/finetune/submit
- ``FinetuneRestart``: POST /api/finetune/job/<job_id>/restart
- ``CreateVolume``: POST /api/finetune/create-volume
- ``LoadCrops``: POST /api/finetune/load-crops
- ``AIAnnotateSettings``, ``AIAnnotateRun``, ``AIAnnotateResend`` and
  ``AIAnnotateDecision``: POST /api/finetune/ai-annotate/{settings,run,resend},
  and accept and reject
- ``BlockwiseValidate``, ``BlockwiseGenerate``, ``BlockwisePrecheck`` and
  ``BlockwiseSubmit``: POST /api/blockwise/{validate,generate,precheck,submit},
  which answer through ``check`` instead (see there)
"""

from typing import Annotated, Any, Optional

from flask import jsonify
from pydantic import (
    AfterValidator,
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


def _number(value, kind, name):
    """``value`` as a number of type ``kind`` (int or float), or a
    ValueError saying ``name`` must be one and what was sent. A whole number
    may come as "25" or 25.0, but not 2.5: int() made that 2."""
    try:
        number = float(value)
    except (TypeError, ValueError):
        number = None
    if number is None or (kind is int and not number.is_integer()):
        raise ValueError(f"{name} must be {'a whole number' if kind is int else 'a number'}, got {value!r}")
    return int(number) if kind is int else number


def _whole_number(value, info: ValidationInfo):
    return _number(value, int, info.field_name)


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


def _voxel_size(value):
    """[z, y, x] in nm from "8", "4,4,8", 8 or [4, 4, 8]; None when blank.
    Whole numbers stay ints: the model's server makes a Coordinate of them."""
    if value is None or (isinstance(value, str) and not value.strip()):
        return None
    parts = value.replace(",", " ").split() if isinstance(value, str) else value
    if not isinstance(parts, (list, tuple)):
        parts = [parts]
    sizes = [_number(p, float, "voxel_size") for p in parts]
    if len(sizes) not in (1, 3) or any(s <= 0 for s in sizes):
        raise ValueError(f"voxel_size must be one positive number or three (z,y,x), got {value!r}")
    sizes = sizes * 3 if len(sizes) == 1 else sizes
    return [int(s) if s.is_integer() else s for s in sizes]


class BioimageSelection(BaseModel):
    """A BioImage Model Zoo model ticked on the Models tab: its id or
    nickname, and the voxel size typed beside it (blank: the model's own)."""

    id: Annotated[str, _required("id", strip=True)] = Field(None, validate_default=True)
    voxel_size: Annotated[Optional[list], BeforeValidator(_voxel_size)] = None


class CellposeSelection(BaseModel):
    """A Cellpose model ticked on the Models tab's Cellpose panel: its name
    (cpsam_v2, cpsam), the voxel size typed beside it, its output, and for
    masks how much two slices' masks must overlap to be linked (IoU; blank or
    0: not linked). A blank voxel size is kept as None here, and refused by
    services.launch with a message naming the model."""

    model: Annotated[str, _required("model", strip=True)] = Field(None, validate_default=True)
    voxel_size: Annotated[Optional[list], BeforeValidator(_voxel_size)] = None
    output: str = "flows"
    stitch_threshold: Annotated[Optional[float], BeforeValidator(
        lambda v: None if v is None or (isinstance(v, str) and not v.strip())
        else _number(v, float, "stitch_threshold"))] = None


class SubmitModels(BaseModel):
    """The Models tab's Submit: the catalog models, Hugging Face repos,
    BioImage Model Zoo and Cellpose models to run. Every other running model
    is stopped (services.launch). ``resample``, when given, becomes the
    session's (``Session.resample``) before the models are started."""

    selected_models: list[str] = []
    selected_hf_models: list[str] = []
    selected_bioimage_models: list[BioimageSelection] = []
    selected_cellpose_models: list[CellposeSelection] = []
    resample: Optional[bool] = None


class Equivalences(BaseModel):
    """A merging postprocessor's merged ids, from its inference server, for
    the segmentation layer whose source ends in ``dataset``: each list is a
    set of ids shown as one."""

    dataset: str
    equivalences: list[list[int]]


# A point or a size, z, y, x.
_Triple = Annotated[list[float], Field(min_length=3, max_length=3)]


class BoundingBox(BaseModel):
    """A box drawn or loaded on an INPUT node: its corner and its size."""

    offset: _Triple = [0, 0, 0]
    shape: _Triple = [1, 1, 1]


class BbxGenerator(BaseModel):
    """The box tool's viewer on ``dataset_path``, showing the boxes the
    INPUT node has; ``num_boxes`` is how many the dialog counts up to."""

    dataset_path: Annotated[str, _required("dataset_path", message="Dataset path is required")] = Field(
        None, validate_default=True)
    num_boxes: WholeNumber = 1
    existing_bounding_boxes: Optional[list[BoundingBox]] = []


# --- The finetune tab's ------------------------------------------------------------


def _form_number(kind, default):
    """A number field of the finetune tab's forms, of type ``kind`` (int or float).

    The form sends what its fields hold, so an emptied field arrives as null
    or "", and is ``default`` here (it reached the trainer as "--num-epochs
    None", which failed only once the job ran). A whole number may come as
    "25" or 25.0, but not 2.5. Anything else is refused with the field's
    name and what was sent.
    """
    def read(value, info: ValidationInfo):
        if value is None or value == "":
            return default
        return _number(value, kind, info.field_name)

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


# --- The finetune tab's AI annotation ----------------------------------------------
#
# The browser picks among what the server's config allows; it never sends an
# endpoint, a project or a credential, so none of these has a field for one.
# Unknown fields are ignored.

# An annotation's id, as the server makes it (ai_annotate.staging's
# ANNOTATE_ID_RE): it is joined into a path, so nothing else is accepted.
_ANNOTATE_ID = Annotated[str, Field(pattern=r"^[0-9a-f]{32}$")]
# The longest prompt and target name the routes take.
PROMPT_MAX_CHARS = 4000
LABEL_MAX_CHARS = 100


def _blank_is_none(value):
    return None if isinstance(value, str) and not value.strip() else value


_Label = Annotated[Optional[Annotated[str, Field(max_length=LABEL_MAX_CHARS)]], BeforeValidator(_blank_is_none)]
# A provider or model id: required, and as short as a label.
_Id = Annotated[str, Field(max_length=LABEL_MAX_CHARS)]


class AIAnnotateSettings(BaseModel):
    """The provider and model to use (ids the config lists), the target, as
    an organelle catalog key and/or a name, the editable prompt (null: the
    catalog's), and whether the user acknowledges where the provider sends
    the images, with the ``destination`` text the page showed them."""

    provider: Annotated[_Id, _required("provider", strip=True)] = Field(None, validate_default=True)
    model: Annotated[_Id, _required("model", strip=True)] = Field(None, validate_default=True)
    label_key: _Label = None
    label_name: _Label = None
    prompt: Optional[str] = Field(None, max_length=PROMPT_MAX_CHARS)
    acknowledge: bool = False
    destination: Optional[str] = Field(None, max_length=1000)


def _finite(point):
    if point is not None and not all(abs(v) < float("inf") for v in point):
        raise ValueError("point_nm must be three finite numbers, z, y, x")
    return point


class AIAnnotateRun(BaseModel):
    """Where to annotate: ``point_nm`` (z, y, x, world nm) in the plane
    normal to ``depth_axis`` (0 z: XY, 1 y: XZ, 2 x: YZ). Either may be left
    out: the view centre, and the viewer layout's plane."""

    point_nm: Annotated[Optional[_Triple], AfterValidator(_finite)] = None
    depth_axis: Optional[Annotated[int, Field(ge=0, le=2)]] = None


class AIAnnotateResend(BaseModel):
    """The staged annotation to ask the model about again, and the edited
    prompt (null or blank: the catalog's)."""

    annotate_id: Annotated[_ANNOTATE_ID, _required("annotate_id")] = Field(None, validate_default=True)
    prompt: Optional[str] = Field(None, max_length=PROMPT_MAX_CHARS)


class AIAnnotateDecision(BaseModel):
    """The staged annotation to accept or reject; on accept, whether its
    labels replace voxels that already have one (``overwrite``) or only
    fill unannotated ones."""

    annotate_id: Annotated[_ANNOTATE_ID, _required("annotate_id")] = Field(None, validate_default=True)
    overwrite: bool = False


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
