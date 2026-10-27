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

import logging
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
    model_validator,
)

from cellmap_flow.blockwise.bounding_box import BoundingBox
from cellmap_flow.finetune.options import MANIFEST_FIELDS, MODES, OutputType, TrainingOptions
from cellmap_flow.jobs.settings import SERVER_CONFIG_KEYS
from cellmap_flow.jobs.site import current_site
from cellmap_flow.models import entry as model_entry


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
    """A model type and its constructor's parameters, as the builder's model
    form sends them; ``registry.build_model`` builds the config from them as
    from any model entry. ``type`` is the type's name (``registry.model_types``,
    "fly"); ``class_name``, the class name ("FlyModelConfig") the form posted
    before 0.3.0, is still taken in its place, with a deprecation warning,
    until the release after."""

    type: Optional[str] = None
    class_name: Optional[str] = None
    params: dict = {}

    @model_validator(mode="after")
    def _one_name(self):
        if not (self.type or self.class_name):
            raise ValueError("type is required")
        return self

    def type_name(self) -> str:
        """The model type to build: ``type``, else the deprecated ``class_name``."""
        if self.type:
            return self.type
        logging.getLogger(__name__).warning(
            f"POST /api/create-model-config: class_name ({self.class_name}) is deprecated and goes in the "
            "release after 0.3.0; post type, the model type's name"
        )
        return self.class_name


class SetData(BaseModel):
    """The dataset to open a new viewer on."""

    dataset_path: Annotated[str, _required("dataset_path", strip=True)] = Field(None, validate_default=True)


def _voxel_size(value):
    """[z, y, x] in nm from "8", "4,4,8", 8 or [4, 4, 8]; None when blank
    (``models.entry.voxel_size``, as the model's constructor reads it)."""
    if model_entry.blank(value):
        return None
    return list(model_entry.voxel_size(value))


class BioimageSelection(BaseModel):
    """A BioImage Model Zoo model ticked on the Models tab: its id or
    nickname, and the voxel size typed beside it (blank: the model's own)."""

    id: Annotated[str, _required("id", strip=True)] = Field(None, validate_default=True)
    voxel_size: Annotated[Optional[list], BeforeValidator(_voxel_size)] = None


class CellposeSelection(BaseModel):
    """A Cellpose model ticked on the Models tab's Cellpose panel: its name
    (cpsam_v2, cpsam), the voxel size typed beside it, its output, whether
    it segments in 3D, and for 2D masks how much two slices' masks must
    overlap to be linked (IoU; blank or 0: not linked). A blank voxel size
    is kept as None here, and refused by services.launch with a message
    naming the model."""

    model: Annotated[str, _required("model", strip=True)] = Field(None, validate_default=True)
    voxel_size: Annotated[Optional[list], BeforeValidator(_voxel_size)] = None
    output: str = "flows"
    do_3d: bool = False
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


class RemoveModel(BaseModel):
    """The ✕ of a model on the Models tab's Loaded models list: the one
    model to stop (services.launch.remove_model), by its name."""

    name: Annotated[str, _required("name", strip=True)] = Field(None, validate_default=True)


class Equivalences(BaseModel):
    """A merging postprocessor's merged ids, from its inference server, for
    the segmentation layer whose source ends in ``dataset``: each list is a
    set of ids shown as one."""

    dataset: str
    equivalences: list[list[int]]


# A point or a size, z, y, x.
_Triple = Annotated[list[float], Field(min_length=3, max_length=3)]


class BbxGenerator(BaseModel):
    """The box tool's viewer on ``dataset_path``, showing the boxes the
    INPUT node has; ``num_boxes`` is how many the dialog counts up to."""

    dataset_path: Annotated[str, _required("dataset_path", message="Dataset path is required")] = Field(
        None, validate_default=True)
    num_boxes: WholeNumber = 1
    existing_bounding_boxes: Optional[list[BoundingBox]] = []


# --- The finetune tab's ------------------------------------------------------------


class _FinetuneBody(TrainingOptions):
    """A body of the Finetune tab's training forms: ``TrainingOptions``' fields, as the
    form sends them. An emptied field arrives as null or "", and takes the default
    (it reached the trainer as "--num-epochs None", which failed only once the job
    ran). A LoRA rank of 0 is ``mode`` full, and the browser form's ``finetune_mode``
    is ``mode`` too. ``output_type`` left blank is detected from the model
    (common.autodetect_output_type). Every other key of the body is kept as sent."""

    model_config = ConfigDict(extra="allow")

    output_type: Optional[OutputType] = Field(None, description=(
        "The training target (TrainingOptions.output_type); blank, the route detects it from the model."
    ))

    @model_validator(mode="before")
    @classmethod
    def _blank_fields_take_the_default(cls, body):
        if isinstance(body, dict):
            body = {key: value for key, value in body.items() if value not in (None, "")}
            if "mode" not in body and body.get("finetune_mode") in MODES:
                body["mode"] = body["finetune_mode"]
        return body

    def changes(self) -> dict:
        """The training settings the body gave, by name (the manifest's aside)."""
        return {
            name: getattr(self, name) for name in TrainingOptions.model_fields
            if name in self.model_fields_set and name not in MANIFEST_FIELDS
        }

    def overrides(self) -> dict:
        """The manifest's settings the body gave (TrainingOptions' MANIFEST_FIELDS): a blank
        one leaves the manifest's value as it is."""
        return {key: getattr(self, key) for key in MANIFEST_FIELDS if key in self.model_fields_set}


class FinetuneSubmit(_FinetuneBody):
    """A new training job: ``model_name`` trained on the corrections at
    ``corrections_path``, a session's corrections directory or the base
    output path of its sessions, with the training settings of
    ``finetune.options.TrainingOptions``, whose defaults are the server
    form's (_finetune_tab.html's values and selected options, checked
    against them by tests/web/test_schema.py; cellmap_flow.schema exports
    them to the browser as FINETUNE_DEFAULTS). The route adjusts them for
    painted annotations and for the model's target (TrainingOptions.for_session).
    """

    model_name: Annotated[str, _required("model_name")] = Field(None, validate_default=True)
    corrections_path: Annotated[str, _required(
        "corrections_path",
        message="corrections_path is required. Please specify the output path where annotation crops are saved.",
    )] = Field(None, validate_default=True)
    checkpoint_path: Optional[str] = None
    auto_serve: bool = True
    queue: Optional[str] = Field(default_factory=lambda: current_site().default_queue)
    charge_group: Optional[str] = None

    def options(self, **resolved) -> TrainingOptions:
        """The settings this body asks for: TrainingOptions with the fields the body gave,
        and ``resolved`` over them (what the route worked out: the output type and
        offsets, the model's output class)."""
        return TrainingOptions.model_validate({**self.changes(), **self.overrides(), **resolved})


class FinetuneRestart(_FinetuneBody):
    """The next iteration of a job: the training settings to change, each checked as
    TrainingOptions checks it (``changes``; the trainer applies them, cli.apply_restart_params),
    and the manifest overrides. Nothing is required: the job's model and corrections
    stay as launched."""


class CreateVolume(BaseModel):
    """A new, empty annotation volume, in a new session under
    ``output_path`` (by default ~/.cellmap_flow/corrections): for
    ``model_name``, on the grid its prediction is served on; or, with no
    model named, a volume painted before any model (a "paint-first" one), on
    the raw's grid at ``voxel_size`` (nm, z, y, x, or one number for all
    three; none: the raw level the viewer shows), which Which model to start
    from's Use this model carries onto a model's grid. A voxel size with a
    model is refused: a model's volume lies on the model's grid."""

    model_name: Annotated[Optional[str], BeforeValidator(
        lambda v: None if v is None or (isinstance(v, str) and not v.strip()) else v)] = None
    voxel_size: Annotated[Optional[list], BeforeValidator(_voxel_size)] = None
    output_path: Any = None

    @model_validator(mode="after")
    def _a_voxel_size_only_without_a_model(self):
        if self.model_name is not None and self.voxel_size is not None:
            raise ValueError(
                f"voxel_size is for a volume made without a model; {self.model_name}'s volume lies on its own grid"
            )
        return self


class LoadCrops(BaseModel):
    """Crops imported into the session's annotation volume: ``yaml`` is the
    YAML's text or its path, ``output_path`` as for CreateVolume, and
    ``load_id`` names the import's progress. ``model_name``, the model
    picked, is only asked for when the session has no volume yet, to lay a
    new one on its grid; without one the new volume is painted before any
    model, as New Volume makes one."""

    yaml: Annotated[str, _required("yaml", message="Missing 'yaml' field")] = Field(None, validate_default=True)
    model_name: Optional[str] = None
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
    # True agrees to send to the provider's destination, False withdraws that,
    # left out leaves it as it is.
    acknowledge: Optional[bool] = None
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
