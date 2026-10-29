"""Every setting a finetune trains with, and its default, in one place.

``TrainingOptions`` is what the Finetune tab's form sends
(``dashboard/requests.FinetuneSubmit`` adds the request's own fields to
it), what the job manager turns into the job's command line
(``cli.training_flags``), what that command line parses back into
(``cli.build_arg_parser`` makes a flag of every field, ``cli.options_of``
reads them), and what the trainer is built from
(``trainer.Finetuner``). Each default is written here once:
``cellmap_flow.schema`` exports the form's to the browser as
``FINETUNE_DEFAULTS``, so the browser's form starts from the same values.

The settings are kept as they are asked for. ``for_session`` says what a
job can train with given its session (sparse annotations, a distance or a
flow target); the trainer applies the rest of its own rules (label
smoothing means nothing to the margin loss, say) when it is built.
"""

import json
from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from cellmap_flow.serving.probe import SIGNED_UNIT, UNBOUNDED, UNIT

__all__ = [
    "TrainingOptions",
    "LOSS_TYPES",
    "OUTPUT_TYPES",
    "MODES",
    "OUTPUT_CLASSES",
    "MANIFEST_FIELDS",
    "LEARNING_RATE_FLOOR",
]

# What --loss-type, --output-type, --mode and --output-class take, here once.
LOSS_TYPES = ("dice", "bce", "combined", "mse", "margin", "interval", "flow")
OUTPUT_TYPES = ("binary", "binary_broadcast", "affinities", "distance", "flows")
MODES = ("lora", "full", "head")
OUTPUT_CLASSES = (UNIT, SIGNED_UNIT, UNBOUNDED)

LossType = Literal["dice", "bce", "combined", "mse", "margin", "interval", "flow"]
OutputType = Literal["binary", "binary_broadcast", "affinities", "distance", "flows"]
Mode = Literal["lora", "full", "head"]
OutputClass = Literal["unit", "signed_unit", "unbounded"]
DistillationScope = Literal["unlabeled", "all"]
LearningRateSchedule = Literal["cosine", "constant"]

# The settings a run writes into its session's manifest rather than onto the
# trainer's command line: the dataset reads them from there (data/dataset.py),
# so the CLI has no flag for them and a restart changes them through the
# manifest (the dashboard's restart route writes it).
MANIFEST_FIELDS = ("patches_per_epoch", "rehearsal_fraction")

# The share of learning_rate a run ends at. A run starts at the form's rate
# and, under the cosine schedule (learning_rate_schedule, the default),
# eases down to this share of it by its last optimizer step, along half a
# cosine: big steps while the model is far from the paint, small ones once
# it is close (trainer/loop.py learning_rate_factor). The constant
# schedule (the form's "Ease down" unticked) keeps the form's rate
# throughout, for a run the user steers by hand. The browser's three
# trainers read it from web/src/generated/python-data.json
# (web/build/export_data.py; web/src/finetune/schedule.ts).
LEARNING_RATE_FLOOR = 0.1


def _is_offset(offset):
    """A [z, y, x] of whole numbers."""
    return (
        isinstance(offset, list)
        and len(offset) == 3
        and all(isinstance(v, int) and not isinstance(v, bool) for v in offset)
    )


class TrainingOptions(BaseModel):
    """The settings of one finetune, each with its default.

    The defaults are the server form's (``_finetune_tab.html``, checked by
    tests/web/test_schema.py): a LoRA adapter of rank 64, 20 epochs of
    batches of 8 from 1e-4 eased down to a tenth of it, the margin loss
    with class balance and a light distillation (0.01) on the unlabeled
    voxels, no augmentation. The browser's form overrides a few for the
    last layer (web/src/finetune/types.ts DEFAULT_OPTIONS).
    """

    model_config = ConfigDict(extra="forbid")

    mode: Mode = Field("lora", description=(
        "What trains: a LoRA adapter on the frozen network (lora), every weight (full), or the "
        "network's last layer (head), which only the browser offers. A LoRA rank of 0 means full."
    ))
    lora_r: int = Field(64, ge=0, description=(
        "LoRA rank; 0 is a full finetune (every parameter trainable, no adapter), which exports "
        "full_finetune/model_state_dict.pt instead of lora_adapter/. Low rank is itself the "
        "anti-forgetting mechanism: the adapter wants just enough capacity to fix the bad regions "
        "and not enough to rewrite the good ones."
    ))
    lora_alpha: Optional[int] = Field(None, ge=0, description=(
        "LoRA alpha, the adapter's scaling (alpha / rank); left blank it is twice the rank, so the "
        "scaling stays 2 whatever the rank."
    ))
    lora_dropout: float = Field(0.1, ge=0.0, lt=1.0, description="LoRA dropout.")
    lora_min_channels: int = Field(0, ge=0, description=(
        "Skip LoRA on layers narrower than this on either side. Narrow full-resolution layers are "
        "where the adapter is expensive and nearly parameter-free: on mito-aff-unet-setup-16, 96 "
        "skips 7 of 19 layers (~1% of adapter params) for a 1.7x faster step. 0 adapts every layer."
    ))
    num_epochs: int = Field(20, ge=1, description="Number of training epochs.")
    batch_size: int = Field(8, ge=1, description="Patches per batch.")
    learning_rate: float = Field(1e-4, gt=0, description=(
        "The learning rate a run starts at (AdamW), which learning_rate_schedule then moves or "
        "keeps; a restart starts at it again."
    ))
    learning_rate_schedule: LearningRateSchedule = Field("cosine", description=(
        "What the learning rate does over a run: 'cosine' eases it down along half a cosine to "
        f"{LEARNING_RATE_FLOOR:g} times it at the run's last step (LEARNING_RATE_FLOOR), big steps "
        "while the model is far from the paint and small ones once it is close; 'constant' keeps "
        "it for every step, as every run did before the schedule."
    ))
    gradient_accumulation_steps: int = Field(1, ge=1, description=(
        "Batches to accumulate gradients over before each optimizer step."
    ))
    loss_type: LossType = Field("margin", description=(
        "The supervised loss. 'interval' is for a distance model (output_type distance) on "
        "scribbles: each painted voxel is held between the bounds the paint implies on its "
        "distance to the boundary, exact where the paint is dense, and the field's slope is "
        "limited (slope_weight). 'flow' is for a flow model such as Cellpose (output_type flows): "
        "MSE of the flows, halved, plus BCE of the foreground, each over the voxels it can be "
        "known at."
    ))
    label_smoothing: float = Field(0.1, ge=0.0, lt=1.0, description=(
        "s moves the targets from 0 and 1 to s/2 and 1 - s/2, which keeps the model from being "
        "pushed to extreme outputs and preserves gradual, distance-like predictions. The margin, "
        "interval and flow losses take none."
    ))
    distillation_lambda: Optional[float] = Field(0.01, ge=0.0, description=(
        "Weight of the distillation term, which holds the model to the one it started from "
        "where the paint says nothing; 0 switches it off. Try 0.5 to 1 for sparse scribbles. "
        "Blank leaves it to the trainer: 1 when the session has good regions, else 0."
    ))
    distillation_scope: DistillationScope = Field("unlabeled", description=(
        "Where the distillation term applies: the unlabeled voxels only, or all voxels. "
        "Good regions, when the session has any, decide instead."
    ))
    margin: float = Field(0.3, ge=0.0, le=0.5, description=(
        "The margin loss's margin: foreground must exceed 1 - margin, background stay below it."
    ))
    balance_classes: bool = Field(True, description=(
        "Weight foreground and background voxels equally, however many of each were painted; "
        "keeps the foreground from being over-predicted."
    ))
    slope_weight: float = Field(1.0, ge=0.0, description=(
        "Weight of the interval loss's slope limit, which keeps the predicted field from getting "
        "steeper than a distance field. Without it the bounds alone let the field collapse into "
        "a step."
    ))
    augment: bool = Field(False, description=(
        "Random flips, XY rotations, brightness and noise on each patch (patch-centre jitter is "
        "always applied). Augmentation pays off when a run revisits the same patches many "
        "times; below a few hundred gradient steps it mostly adds variance, which is why it is "
        "off by default."
    ))
    mixed_precision: bool = Field(True, description=(
        "Mixed precision on CUDA: bf16 where the GPU has it, else fp16 with a GradScaler, falling "
        "back to fp32 when the model NaNs under either. Off on other devices."
    ))
    num_workers: int = Field(4, ge=0, description="DataLoader workers.")
    tensorboard: bool = Field(True, description=(
        "Write TensorBoard event files to <output-dir>/tensorboard "
        "(view with `tensorboard --logdir <training dir>`)."
    ))
    mask_unannotated: bool = Field(False, description=(
        "Sparse annotations: only annotated voxels count in the loss (0 unannotated, "
        "1 background, 2 and up foreground). The dashboard sets it when the session is painted."
    ))
    output_type: OutputType = Field("binary", description=(
        "How training targets are made from the annotations. 'binary': single-channel fg/bg "
        "(with select_channel for multi-channel models). 'binary_broadcast': the binary target "
        "on every output channel. 'affinities': affinity targets from instance labels (needs "
        "offsets). 'distance': the soft signed-distance target (tanh(d/sigma)+1)/2 of the "
        "fly_organelles and cellmap *_distance_* models; needs loss_type bce, or interval for "
        "scribbles. 'flows': per-slice flows towards each painted instance's centre and a "
        "foreground channel, for a flow model such as Cellpose; needs loss_type flow."
    ))
    distance_sigma: float = Field(6.0, gt=0, description=(
        "tanh scale in output voxels of the distance target. The cellmap distance models were "
        "trained with 6."
    ))
    select_channel: Optional[int] = Field(None, ge=0, description=(
        "The one channel of a multi-channel model to train on the paint (not with output_type "
        "affinities or flows, whose targets have a channel for each of the model's); every other "
        "channel is held to the starting model's prediction on every voxel, so it stays as it was. "
        "Blank trains every channel on the one target."
    ))
    offsets: Optional[str] = Field(None, description=(
        "JSON list of [dz, dy, dx] offsets for affinity targets, e.g. '[[1,0,0],[0,1,0],[0,0,1]]'. "
        "Blank with output_type affinities reads 'offsets' from the model script."
    ))
    output_class: Optional[OutputClass] = Field(None, description=(
        "What the model's output is, as its server reports it (serving.probe): unit for a "
        "sigmoid baked in ([0, 1]), signed_unit for a tanh ([-1, 1]), unbounded for logits. The "
        "losses then compare probabilities or logits accordingly, as ranking and seeding read "
        "the same output. Blank, the training job probes the model once."
    ))
    patches_per_epoch: Optional[int] = Field(None, description=(
        "Patches drawn per epoch; blank or 0 draws one per populated chunk. Kept in the "
        "session's manifest."
    ))
    rehearsal_fraction: Optional[float] = Field(None, ge=0.0, le=1.0, description=(
        "The share of each epoch's patches drawn from the good regions, 0 to 1; 0 turns "
        "rehearsal off without dropping the regions. Blank keeps the manifest's value."
    ))

    @field_validator("patches_per_epoch", mode="before")
    @classmethod
    def _count_or_auto(cls, value):
        if value is None:
            return None
        try:
            count = int(value)
        except (TypeError, ValueError):
            count = -1
        if count < 0:
            raise ValueError("patches_per_epoch must be a non-negative integer")
        return None if count == 0 else count

    @field_validator("offsets", mode="before")
    @classmethod
    def _offsets_as_json(cls, value):
        """The offsets as the JSON the trainer's flag takes; a list is dumped, a string checked."""
        if value is None:
            return None
        offsets = json.loads(value) if isinstance(value, str) else value
        if not (isinstance(offsets, list) and offsets and all(_is_offset(o) for o in offsets)):
            raise ValueError("offsets must be a list of [z, y, x] integer offsets")
        return json.dumps(offsets)

    @model_validator(mode="after")
    def _mode_and_rank_agree(self):
        """A rank of 0 is a full finetune, and a full finetune has no adapter: lora_r and
        lora_alpha are 0 (as the job's record keeps them); otherwise alpha is twice the rank
        when not given."""
        if self.mode == "lora" and self.lora_r == 0:
            self.__dict__["mode"] = "full"
        if self.mode == "full":
            self.__dict__["lora_r"] = 0
            self.__dict__["lora_alpha"] = 0
        elif self.lora_alpha is None:
            self.__dict__["lora_alpha"] = 2 * self.lora_r
        return self

    def replaced(self, **changes) -> "TrainingOptions":
        """A copy with ``changes`` applied, validated as a whole."""
        return TrainingOptions.model_validate({**self.model_dump(), **changes})

    def for_session(self, *, sparse: bool) -> "TrainingOptions":
        """What a job trains with, given whether its session is sparse (painted).

        Some combinations cannot train, or would train the wrong thing, so
        they are replaced here, where the dashboard shows it in submit's
        answer, rather than failing on the cluster:
        - sparse annotations with mse: the margin loss, with distillation to
          the base model at 0.5, which is how sparse annotations train (see
          the distance case below);
        - a distance target with sparse annotations: the interval loss, on the
          bounds the paint sets on each voxel's distance, with distillation of
          at least 0.5 (below);
        - a distance target otherwise: bce, without label smoothing;
        - a flow target (Cellpose): the flow loss, the only one it goes with,
          without label smoothing, sparse or not (it masks per channel itself).
        A sparse session also masks its unannotated voxels out of the loss.
        """
        changes = {"mask_unannotated": bool(sparse)}
        if self.output_type == "flows":
            changes.update(loss_type="flow", label_smoothing=0.0)
        else:
            if sparse and self.loss_type == "mse":
                changes.update(loss_type="margin", distillation_lambda=0.5)
            if self.output_type == "distance" and sparse:
                # A distance target needs the 3D object boundary, and scribbles
                # are strokes with unannotated voxels all around them: the exact
                # target would supervise next to nothing. They do bound each
                # painted voxel's distance, between the nearest voxel not painted
                # as its own class and the nearest painted as the other, and the
                # interval loss asks only for that (target_transforms.
                # IntervalTargetTransform, losses.IntervalLoss). Where the paint
                # is dense the two meet and the distance is exact.
                #
                # Distillation of at least 0.5 holds the unpainted voxels, which
                # get no bounds: the form's default, 0.01, held almost nothing.
                changes.update(
                    loss_type="interval", label_smoothing=0.0,
                    distillation_lambda=max(self.distillation_lambda or 0.0, 0.5),
                )
            elif self.output_type == "distance":
                # The soft distance target is only defined against BCE-with-logits;
                # margin and dice assume hard labels, and smoothing would blur a
                # target that is already soft.
                changes.update(loss_type="bce", label_smoothing=0.0)
        return self.replaced(**changes)
