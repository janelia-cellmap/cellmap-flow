"""``FlyModelConfig``: a network fly_organelles trained, from its checkpoint file.

``checkpoint_path`` is one of

- a training checkpoint (``model_checkpoint_<iteration>``): the state dict
  fly_organelles' training saves, loaded into its ``StandardUnet``, whose
  architecture is read from the shapes of the weights (``unet_arguments``);
- a TorchScript file (``.ts``);
- a whole pickled model (``model.pt``), which is unpickled only when
  ``CELLMAP_FLOW_ALLOW_PICKLE`` allows it.

A folder that cellmap_models exported (metadata.json and model.ts) is the
``cellmap`` type's: it is served as exported, at the tile it was exported
at, and given here as ``checkpoint_path`` it is refused with a pointer to
``type: cellmap``. The ``model.pt`` in such a folder can still be served
here, at another ``input_size``.

What the constructor is not given, it reads from the files beside the
checkpoint (``run_metadata``); an argument given always wins:

- ``metadata.json``, when the file is in a cellmap_models export folder:
  channel names, voxel sizes, input and output size;
- ``config.yaml``, the run configuration fly_organelles' ``load_config``
  reads: ``run.labels``, ``run.voxel_size``, ``checkpoint.input_shape`` and
  ``output_shape``;
- ``snapshots/``, the gunpowder snapshots fly_organelles' training writes:
  the ``raw`` and ``output`` arrays' shapes and voxel sizes, and so the
  number of channels the network outputs;
- ``train.py``, the training script of a run: its top-level ``labels`` and
  ``voxel_size`` assignments, when they are literals. The script is parsed,
  never run.

Labels name the channels only when there is one per channel the snapshots
show: an affinity or LSD run's network outputs several per label. The voxel
sizes are read only when neither is given, and the sizes only when neither
is: an output size belongs to the input size it came with. Then, when
something is still missing: one voxel size stands for the other
(fly_organelles trains at a single voxel size); no input size is
fly_organelles' training tile, 178 voxels a side; no output size is
computed from the network when it is built. Channels and a voxel size have
no default: without them the constructor raises, naming the files it reads.

``sigmoid`` (True by default): the network's output goes through a sigmoid,
which is added unless the network ends in one already, as a cellmap_models
export's model.pt and model.ts do. fly_organelles trains its networks on
logits (its BCE, focal and Dice losses apply the sigmoid themselves), so a
training checkpoint gets one. False serves the network's output as it is.

A training checkpoint or a model.pt needs fly_organelles, so those run in
pixi.toml's ``fly`` environment unless the entry gives another ``env``; a
TorchScript file runs in any. torch and fly_organelles are imported only
when the model is built: the CLIs, ``--help`` and the dashboard's model form
import every type.
"""

import ast
import json
import logging
import os
import re

import numpy as np
import yaml

from cellmap_flow.models.configs.base import Config, ModelConfig, _as_int_tuple, _get_device
from cellmap_flow.models.geometry import _numbers

logger = logging.getLogger(__name__)

# fly_organelles' training tile: the input_shape its load_config and run()
# default to. A StandardUnet with the default 2x2x2 pooling takes it.
TRAINING_TILE = 178

TORCHSCRIPT, PICKLE, STATE_DICT = "torchscript", "pickle", "state dict"


def checkpoint_format(checkpoint_path: str) -> str:
    """How ``checkpoint_path`` is loaded: TORCHSCRIPT, PICKLE or STATE_DICT."""
    if checkpoint_path.endswith(".ts"):
        return TORCHSCRIPT
    if checkpoint_path.endswith("model.pt"):
        return PICKLE
    return STATE_DICT


def _voxel_size(value):
    """A voxel size given as one number, "16,16,16" or one per axis: three numbers, ints kept ints.

    Not cellpose's alike: importing that module here would define its type
    before this one, and the registry lists the types in definition order.
    """
    if isinstance(value, str):
        value = [float(v) for v in value.replace("(", "").replace(")", "").split(",") if v.strip()]
    if np.ndim(value) == 0:
        value = [value] * 3
    elif len(value) == 1:
        value = list(value) * 3
    return _numbers(float(v) for v in value)


def _flag(value) -> bool:
    """A bool from True/False, or the "false" the server CLI and model form send."""
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered not in ("true", "false", "1", "0", "yes", "no"):
            raise ValueError(f"sigmoid must be true or false, not {value!r}")
        return lowered in ("true", "1", "yes")
    return bool(value)


# --- what the files beside a checkpoint say ------------------------------------

def _from_metadata_json(run_dir):
    """A cellmap_models export's metadata.json: its channel names, voxel sizes and tile."""
    with open(os.path.join(run_dir, "metadata.json")) as f:
        metadata = json.load(f)
    found = {}
    if metadata.get("channels_names"):
        found["channels"] = list(metadata["channels_names"])
    for key in ("input_voxel_size", "output_voxel_size"):
        if metadata.get(key) is not None:
            found[key] = metadata[key]
    # input_shape is (batch, channel, z, y, x) in the exports, output_shape
    # (z, y, x); the last three are the tile either way.
    if metadata.get("input_shape") and metadata.get("output_shape"):
        found["sizes"] = (metadata["input_shape"][-3:], metadata["output_shape"][-3:])
    return found


def _from_run_config(run_dir):
    """fly_organelles' run configuration, the config.yaml its ``config.load_config`` reads."""
    with open(os.path.join(run_dir, "config.yaml")) as f:
        config = yaml.safe_load(f) or {}
    run = config.get("run") or {}
    checkpoint = config.get("checkpoint") or {}
    found = {}
    # An LSD run's network outputs 13 channels per label (validate_run).
    if run.get("labels") and not run.get("lsd"):
        found["labels"] = list(run["labels"])
    if run.get("voxel_size") is not None:
        found["input_voxel_size"] = found["output_voxel_size"] = run["voxel_size"]
    if checkpoint.get("input_shape") and checkpoint.get("output_shape"):
        found["sizes"] = (checkpoint["input_shape"], checkpoint["output_shape"])
    return found


def _zarr_array(path):
    """(shape, voxel_size) of a zarr v2 array, from its .zarray and .zattrs."""
    with open(os.path.join(path, ".zarray")) as f:
        shape = json.load(f)["shape"]
    with open(os.path.join(path, ".zattrs")) as f:
        voxel_size = json.load(f).get("voxel_size")
    return shape, voxel_size


def _from_snapshots(run_dir):
    """The newest training snapshot's ``raw`` and ``output``: the tile, voxel sizes and channels.

    fly_organelles' training snapshots a batch every few thousand
    iterations as ``snapshots/<iteration>.zarr``, each array (batch,
    channel, z, y, x): ``raw`` is the network's input, ``output`` what it
    made of it.
    """
    folder = os.path.join(run_dir, "snapshots")
    snapshots = sorted(name for name in os.listdir(folder) if name.endswith(".zarr"))
    if not snapshots:
        return {}
    snapshot = os.path.join(folder, snapshots[-1])
    raw_shape, raw_voxel_size = _zarr_array(os.path.join(snapshot, "raw"))
    output_shape, output_voxel_size = _zarr_array(os.path.join(snapshot, "output"))
    found = {"sizes": (raw_shape[-3:], output_shape[-3:])}
    if len(output_shape) >= 4:
        found["out_channels"] = int(output_shape[-4])
    if raw_voxel_size is not None:
        found["input_voxel_size"] = raw_voxel_size
    if output_voxel_size is not None:
        found["output_voxel_size"] = output_voxel_size
    return found


def _from_train_script(run_dir):
    """A training script's top-level ``labels = [...]`` and ``voxel_size = ...``, when literals.

    The script is parsed, not run: run, it trains. The last assignment of
    each counts, as it would when the script runs; one that is computed
    rather than written out leaves that value unknown.
    """
    with open(os.path.join(run_dir, "train.py")) as f:
        tree = ast.parse(f.read())
    found = {}
    for node in tree.body:
        if not (isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name)):
            continue
        name = node.targets[0].id
        keys = {"labels": ("labels",), "voxel_size": ("input_voxel_size", "output_voxel_size")}.get(name)
        if keys is None:
            continue
        for key in keys:
            found.pop(key, None)
        try:
            value = ast.literal_eval(node.value)
        except (ValueError, TypeError, SyntaxError):
            continue
        if name == "labels":
            if isinstance(value, (list, tuple)) and value and all(isinstance(v, str) for v in value):
                found["labels"] = list(value)
        else:
            found["input_voxel_size"] = found["output_voxel_size"] = value
    return found


# Read in this order; for each value, the first that has it is used.
_SOURCES = (
    ("metadata.json", _from_metadata_json),
    ("config.yaml", _from_run_config),
    ("snapshots", _from_snapshots),
    ("train.py", _from_train_script),
)


def run_metadata(checkpoint_path: str) -> dict:
    """What the files beside ``checkpoint_path`` say about its network, as {key: (value, file)}.

    Keys: ``channels``, ``input_voxel_size``, ``output_voxel_size`` and
    ``sizes``, the input and output size as a pair: an output size belongs
    to the input size it came with. Each from the first file that has it,
    of metadata.json, config.yaml, snapshots/ and train.py (see the module
    docstring). A file that is missing is skipped, and one that cannot be
    read is skipped with a warning: it should not stop a model whose
    arguments say it all.
    """
    run_dir = os.path.dirname(os.path.abspath(checkpoint_path))
    found = {}
    for source, read in _SOURCES:
        try:
            values = read(run_dir)
        except FileNotFoundError:
            continue
        except (OSError, ValueError, SyntaxError, KeyError, IndexError, TypeError, yaml.YAMLError) as e:
            logger.warning(f"Not reading {os.path.join(run_dir, source)}: {e}")
            continue
        for key, value in values.items():
            found.setdefault(key, (value, source))

    # Training labels name the channels only when there is one per channel
    # the network outputs, as far as the snapshots say.
    out_channels = found.pop("out_channels", (None,))[0]
    if "labels" in found:
        labels, source = found.pop("labels")
        if out_channels is not None and out_channels != len(labels):
            logger.warning(
                f"{source} trains {len(labels)} label(s) {labels}, but the network outputs "
                f"{out_channels} channels: not using them as channel names"
            )
        else:
            found.setdefault("channels", (labels, source))
    return found


# --- the network ---------------------------------------------------------------

_DOWN_CONV = re.compile(r"unet_backbone\.l_conv\.(\d+)\.conv_pass\.(\d+)\.weight")


def unet_arguments(state_dict) -> dict:
    """fly_organelles ``StandardUnet``'s constructor arguments, read from its weights' shapes.

    The output channels are the final convolution's; the levels, feature
    maps and kernel sizes the down path's. The pooling has no weights, so
    the downsample factors are StandardUnet's default, 2x2x2 at each level:
    a network pooled otherwise loads, but its output is not the size
    declared, which the shape check reports.
    """
    convs = {}
    for key, value in state_dict.items():
        match = _DOWN_CONV.fullmatch(key)
        if match:
            convs.setdefault(int(match[1]), {})[int(match[2])] = tuple(value.shape)
    if not convs or "final_conv.weight" not in state_dict:
        raise ValueError(
            "This is not a fly_organelles StandardUnet state dict: it has no "
            "unet_backbone.l_conv or final_conv weights"
        )
    # Each level's convolutions in order, each weight (out, in, *kernel).
    levels = [[shape for _, shape in sorted(convs[level].items())] for level in sorted(convs)]
    fmaps = [level[0][0] for level in levels]
    inc_factor = fmaps[1] / fmaps[0] if len(fmaps) > 1 else 1
    return {
        "out_channels": int(state_dict["final_conv.weight"].shape[0]),
        "num_fmaps": fmaps[0],
        "fmap_inc_factor": int(inc_factor) if float(inc_factor).is_integer() else inc_factor,
        "downsample_factors": [(2, 2, 2)] * (len(levels) - 1),
        "kernel_size_down": [[shape[2:] for shape in level] for level in levels],
    }


def _ends_in_sigmoid(module) -> bool:
    """Whether ``module``'s last step is a sigmoid: it is one, or a Sequential ending in one.

    Only a Sequential's last module is its last step; any other module's is
    whatever its forward does. TorchScript modules keep their class's name
    in ``original_name``.
    """
    while True:
        kind = getattr(module, "original_name", None) or type(module).__name__
        if kind == "Sigmoid":
            return True
        children = list(module.children()) if kind == "Sequential" else []
        if not children:
            return False
        module = children[-1]


class FlyModelConfig(ModelConfig):
    """A network fly_organelles trained, from a training checkpoint, a .ts or a model.pt.

    Args:
        checkpoint_path: the checkpoint file (see the module docstring).
        channels: the name of each output channel. Default: read beside the
            checkpoint.
        input_voxel_size, output_voxel_size: nm per voxel (one number or one
            per axis). Default: read beside the checkpoint; one stands for
            the other.
        input_size, output_size: voxels a side of a tile in and out. Default:
            read beside the checkpoint, else 178 in and the output computed
            from the network. An output size needs its input size.
        sigmoid: pass the output through a sigmoid, added unless the network
            ends in one (default True); False serves the network's output.
    """

    cli_name = "fly"

    finetunable = True

    def __init__(
        self,
        checkpoint_path: str,
        channels: list[str] = None,
        input_voxel_size: tuple = None,
        output_voxel_size: tuple = None,
        name: str = None,
        input_size=None,
        output_size=None,
        scale=None,
        sigmoid: bool = True,
    ):
        super().__init__()
        self.checkpoint_path = str(checkpoint_path)
        self.name = name
        self.scale = scale
        self.sigmoid = _flag(sigmoid)
        self._model = None
        # Set when a training checkpoint is loaded, to compute its output size.
        self._unet_arguments = None
        self._refuse_a_folder()

        found = run_metadata(self.checkpoint_path)
        used = []

        def given_or_found(key, given):
            if given is not None or key not in found:
                return given
            value, source = found[key]
            used.append(f"{key} {value} from {source}")
            return value

        # The server CLI passes the channels as "mito,er".
        if isinstance(channels, str):
            channels = [c.strip() for c in channels.split(",") if c.strip()]
        channels = given_or_found("channels", channels)
        # The voxel sizes and the sizes are each a pair: one given stands for
        # its partner rather than letting the files supply it, since the files
        # describe the network at the other one.
        if input_voxel_size is None and output_voxel_size is None:
            input_voxel_size = given_or_found("input_voxel_size", None)
            output_voxel_size = given_or_found("output_voxel_size", None)
        if output_size is not None and input_size is None:
            raise ValueError(
                "FlyModelConfig got output_size but no input_size: give the input size it comes "
                "with, or neither for the training tile"
            )
        if input_size is None and "sizes" in found:
            (input_size, output_size), source = found["sizes"]
            used.append(f"input_size {list(input_size)} and output_size {list(output_size)} from {source}")
        if used:
            logger.info(f"Fly model {self.name or self.checkpoint_path}: {', '.join(used)}")

        if not channels:
            raise ValueError(self._not_found("channels"))
        if input_voxel_size is None and output_voxel_size is None:
            raise ValueError(self._not_found("input_voxel_size"))
        self.channels = list(channels)
        # fly_organelles trains at one voxel size, in and out. Not
        # _as_int_tuple: that truncated a 5.24 nm voxel to 5.
        self.input_voxel_size = _voxel_size(input_voxel_size if input_voxel_size is not None else output_voxel_size)
        self.output_voxel_size = _voxel_size(output_voxel_size if output_voxel_size is not None else input_voxel_size)
        # The server CLI passes these as "178,178,178" strings.
        self.input_size = _as_int_tuple(input_size)
        self.output_size = _as_int_tuple(output_size)

    def _not_found(self, argument):
        return (
            f"FlyModelConfig for {self.checkpoint_path} needs {argument}: give it, or put beside the "
            "checkpoint a file that says it (metadata.json, config.yaml, snapshots/ or train.py; "
            "none there does)"
        )

    def _refuse_a_folder(self):
        """Raise for a folder: this type takes the checkpoint file, and cellmap serves exports."""
        if not os.path.isdir(self.checkpoint_path):
            return
        if os.path.exists(os.path.join(self.checkpoint_path, "metadata.json")):
            raise ValueError(
                f"{self.checkpoint_path} is a folder cellmap_models exported (it has a metadata.json): "
                f"serve it with type: cellmap, folder_path: {self.checkpoint_path}. type: fly takes a "
                "checkpoint file, such as the model.pt in it to serve at another input_size."
            )
        raise ValueError(
            f"checkpoint_path {self.checkpoint_path} is a folder: give the checkpoint file in it "
            "(model_checkpoint_<iteration>, a .ts or model.pt)"
        )

    @property
    def default_env(self):
        """``fly`` for a training checkpoint or an eager model.pt, else none (TorchScript runs anywhere).

        A training checkpoint is loaded into fly_organelles' ``StandardUnet``,
        and unpickling an eager ``model.pt`` imports fly_organelles' classes:
        pixi.toml's ``fly`` environment has them, cellmap-flow's own does
        not. A ``.ts`` file carries its own code and runs wherever torch does.
        """
        return None if checkpoint_format(self.checkpoint_path) == TORCHSCRIPT else "fly"

    def _load_network(self):
        """The checkpoint's network, on the device, before any sigmoid is added."""
        # Imported here rather than at module scope: importing torch costs
        # ~7s, and the CLI builds its command list from this module, so
        # `cellmap_flow --help` paid that before printing anything.
        import torch

        device = _get_device()
        path = self.checkpoint_path
        kind = checkpoint_format(path)
        if kind == TORCHSCRIPT:
            return torch.jit.load(path, map_location=device)
        if kind == PICKLE:
            # torch.load(weights_only=False) unpickles arbitrary objects, which
            # is a code-exec sink if the checkpoint comes from an untrusted
            # location. Require the operator to opt-in explicitly.
            trusted = os.environ.get("CELLMAP_FLOW_ALLOW_PICKLE", "").lower() in ("1", "true", "yes")
            if not trusted:
                raise ValueError(
                    f"Refusing to torch.load (weights_only=False) checkpoint {path}. "
                    "This unpickles arbitrary objects. If the checkpoint is from a trusted "
                    "source, set CELLMAP_FLOW_ALLOW_PICKLE=1 in the environment."
                )
            return torch.load(path, weights_only=False, map_location=device)

        from fly_organelles.model import StandardUnet

        # mmap: a training checkpoint holds the optimizer's state too, twice
        # the weights' size (9.5 GB for 3.2 GB of weights), and only the
        # weights are read.
        checkpoint = torch.load(path, weights_only=True, map_location="cpu", mmap=True)
        state_dict = checkpoint.get("model_state_dict", checkpoint)
        arguments = unet_arguments(state_dict)
        if arguments["out_channels"] != len(self.channels):
            raise ValueError(
                f"{path} outputs {arguments['out_channels']} channels, but {len(self.channels)} "
                f"are named ({', '.join(self.channels)}): give one name per channel"
            )
        network = StandardUnet(**arguments)
        network.load_state_dict(state_dict)
        self._unet_arguments = arguments
        return network.to(device)

    def load_eval_model(self):
        """The network to serve, in eval mode: the checkpoint's, with a sigmoid if ``sigmoid``."""
        import torch

        network = self._load_network()
        if checkpoint_format(self.checkpoint_path) == STATE_DICT:
            # In a Sequential either way, as fly_organelles' load_eval_model
            # builds it: a LoRA adapter trained on it is keyed by its module
            # names, "0.unet_backbone...".
            layers = [network, torch.nn.Sigmoid()] if self.sigmoid else [network]
            model = torch.nn.Sequential(*layers)
        elif self.sigmoid and not _ends_in_sigmoid(network):
            model = torch.nn.Sequential(network, torch.nn.Sigmoid())
        else:
            if self.sigmoid:
                logger.info(f"{self.checkpoint_path} ends in a sigmoid of its own; none added")
            model = network
        model.to(_get_device())
        model.eval()
        return model

    @property
    def model(self):
        if self._model is None:
            self._model = self.load_eval_model()
        return self._model

    def _computed_output_size(self, input_size):
        """The network's output size for ``input_size``, run once to see.

        A StandardUnet is run on the meta device, which computes shapes and
        nothing else; any other network on zeros, as it is.
        """
        import torch

        try:
            with torch.no_grad():
                if self._unet_arguments is not None:
                    from fly_organelles.model import StandardUnet

                    with torch.device("meta"):
                        output = StandardUnet(**self._unet_arguments)(torch.empty(1, 1, *input_size))
                else:
                    weight = next(self.model.parameters(), None)
                    device = weight.device if weight is not None else "cpu"
                    output = self.model(torch.zeros(1, 1, *input_size, device=device))
        except RuntimeError as e:
            raise ValueError(
                f"{self.checkpoint_path} does not take an input of {list(input_size)} voxels ({e}). Give an "
                "input_size it takes (178 + 16k for fly_organelles' StandardUnet), or with its output_size."
            ) from e
        output_size = tuple(int(s) for s in output.shape[2:])
        logger.info(f"{self.checkpoint_path}: {list(input_size)} voxels in make {list(output_size)} out")
        return output_size

    def _get_config(self):
        config = Config()
        config.model = self.model
        input_size = self.input_size or (TRAINING_TILE,) * 3
        output_size = self.output_size or self._computed_output_size(input_size)
        # Plain numbers rather than Coordinates, which are integers: a
        # 5.24 nm z would be 5, and every chunk placed on the wrong grid.
        config.input_voxel_size = self.input_voxel_size
        config.output_voxel_size = self.output_voxel_size
        config.read_shape = _numbers(np.array(input_size) * np.asarray(self.input_voxel_size, dtype=float))
        # Output voxels are output_voxel_size wide; using the input voxel size
        # here made the context and the shape check wrong whenever they differ.
        config.write_shape = _numbers(np.array(output_size) * np.asarray(self.output_voxel_size, dtype=float))
        config.channels = self.channels
        config.output_channels = len(self.channels)
        config.block_shape = np.array(tuple(output_size) + (config.output_channels,))
        config.output_dtype = np.float32
        return config

    def to_dict(self):
        """This config as a model entry, which ``registry.build_model`` rebuilds.

        With what was read beside the checkpoint, so the entry says what is
        served; the sizes when given or read, not when computed. In the
        constructor's order, which exported YAMLs keep.
        """
        result = {
            "type": "fly",
            "checkpoint_path": self.checkpoint_path,
            "channels": list(self.channels),
            "input_voxel_size": list(self.input_voxel_size),
            "output_voxel_size": list(self.output_voxel_size),
        }
        if self.name is not None:
            result["name"] = self.name
        if self.input_size is not None:
            result["input_size"] = list(self.input_size)
        if self.output_size is not None:
            result["output_size"] = list(self.output_size)
        if self.scale is not None:
            result["scale"] = self.scale
        if not self.sigmoid:
            result["sigmoid"] = False
        return result
