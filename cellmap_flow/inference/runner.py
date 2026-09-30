"""Running a model on a region: the device, the warmup and the forward.

``ModelRunner`` puts a ModelConfig's model on the device, warms it up (the
same forward checks the declared shapes and probes the output's range), and
predicts an output region from an ImageDataInterface: the model's raw output,
``(C, *spatial)``, with no postprocessing and no process-wide state.
``cellmap_flow.inferencer.Inferencer`` adds the normalization and
postprocessing chain on top.

``DeviceSlots`` is how many chunks may use the device at once, and in what
order; the inference server passes one, blockwise workers none.

Importing this module is cheap: torch is imported when it is first needed.
"""

import collections
import contextlib
import logging
import os
import threading
import time

import numpy as np
from funlib.geometry import Coordinate

logger = logging.getLogger(__name__)

GPU_SLOTS_ENV = "CELLMAP_FLOW_GPU_SLOTS"

# Half precision is kept only if its output stays within this fraction of
# the fp32 output's range (1% is under 3 levels of a uint8 display).
HALF_PRECISION_TOLERANCE = 0.01


class ChunkCancelled(Exception):
    """The chunk was not computed: whoever asked for it stopped waiting."""


class DeviceSlots:
    """Lets at most ``n`` chunks use the device at once, in the order they asked.

    The inference server answers every request on its own thread, and
    neuroglancer asks for 6 chunks at once over HTTP/1.1, or dozens through an
    HTTP/2 proxy. Unbounded, k concurrent forwards interleave on the GPU and
    finish together, after about k forwards' time, so the first chunks of a
    view all appeared at once, late. Each forward also holds its own
    activations, and a dozen at once ran an 11 GB card out of memory. One
    slot makes the GPU compute chunks one after another, the first one asked
    for first, at the same total throughput: reading and normalizing the next
    chunk, which stay outside the slot, overlap the current forward.

    Strictly first come, first served: a thread that arrives while a slot is
    free still waits behind anyone already waiting. A plain Semaphore lets
    newcomers barge ahead.

    The server takes ``n`` from ``CELLMAP_FLOW_GPU_SLOTS`` (default 1) when it
    starts. More than one slot only helps a model too small to keep the
    device busy on its own.

    ``hold(cancelled)`` asks ``cancelled()`` while it waits, and once more just
    before taking a slot, and raises ChunkCancelled instead of taking it once
    that says yes. Neuroglancer drops the requests for chunks that have left
    the view after a pan; without this, those still ran their forward ahead
    of the new view's chunks. A chunk already on the device finishes.
    """

    # How often a waiting request asks whether its client is still there.
    POLL_SECONDS = 0.1

    def __init__(self, n=1):
        n = int(n)
        if n < 1:
            raise ValueError(f"DeviceSlots needs at least 1 slot, got {n}")
        self.n = n
        self._cond = threading.Condition()
        self._waiting = collections.deque()  # one ticket per waiting thread, oldest first
        self._running = 0

    @classmethod
    def from_env(cls):
        value = os.environ.get(GPU_SLOTS_ENV, "1")
        try:
            return cls(int(value))
        except ValueError:
            raise ValueError(
                f"{GPU_SLOTS_ENV} must be a whole number of at least 1, got {value!r}"
            ) from None

    @contextlib.contextmanager
    def hold(self, cancelled=None):
        ticket = object()
        with self._cond:
            self._waiting.append(ticket)
            try:
                while self._waiting[0] is not ticket or self._running >= self.n:
                    if cancelled is not None and cancelled():
                        raise ChunkCancelled()
                    self._cond.wait(self.POLL_SECONDS if cancelled else None)
                if cancelled is not None and cancelled():
                    raise ChunkCancelled()
            except BaseException:
                self._waiting.remove(ticket)
                self._cond.notify_all()  # the one behind may be next now
                raise
            self._waiting.popleft()
            self._running += 1
            # With more than one slot free, the next in line may go too.
            self._cond.notify_all()
        try:
            yield
        finally:
            with self._cond:
                self._running -= 1
                self._cond.notify_all()


def _device_part(device_slots, cancelled=None):
    if device_slots is None:
        return contextlib.nullcontext()
    return device_slots.hold(cancelled)


def _autocast(device, dtype):
    if dtype is None:
        return contextlib.nullcontext()
    import torch

    return torch.autocast(device.type, dtype=dtype)


def predict(read_roi, write_roi, config, **kwargs):
    """The default forward: read ``read_roi`` from ``kwargs["idi"]``, run
    ``config.model`` on ``kwargs["device"]``, return ``(C, *spatial)``.

    A script may define its own ``predict`` with this signature instead; it
    is then given ``idi``, ``device`` and ``use_half_prediction``.
    """
    import torch

    idi = kwargs.get("idi")
    if idi is None:
        raise ValueError("idi must be provided in kwargs")

    device = kwargs.get("device")
    if device is None:
        raise ValueError("device must be provided in kwargs")

    use_half_prediction = kwargs.get("use_half_prediction", False)

    raw_input = idi.to_ndarray_ts(read_roi)
    raw_input = np.expand_dims(raw_input, (0, 1))

    # Only the transfer, the forward and the copy back take a device slot;
    # the read and the normalization above overlap another chunk's forward.
    autocast_dtype = kwargs.get("autocast_dtype")
    with _device_part(kwargs.get("device_slots"), kwargs.get("cancelled")), torch.no_grad():
        raw_input_torch = torch.from_numpy(raw_input).to(device, non_blocking=True)
        logger.debug(f"Predicting with model {type(config.model).__name__} on device {device}")
        logger.debug(f"Input shape: {raw_input_torch.shape}, dtype: {raw_input_torch.dtype}")
        raw_input_torch = raw_input_torch.half() if use_half_prediction else raw_input_torch.float()
        with _autocast(device, autocast_dtype):
            output = config.model.forward(raw_input_torch)
        if autocast_dtype is not None:
            output = output.float()  # what the postprocessing always got
        result = output.cpu().numpy()[0]
        logger.debug(f"Output shape: {result.shape}, dtype: {result.dtype}")
    return result


class ModelRunner:
    def __init__(
        self,
        model_config,
        use_half_prediction=False,
        device_slots=None,
        half_precision=False,
    ):
        """Build ``model_config``'s model, put it on the device and warm it up.

        ``device_slots``: a DeviceSlots bounding how many chunks use the
        device at once, as the inference server passes. ``None``, as blockwise
        workers (one chunk at a time) use, bounds nothing.

        ``half_precision``, or ``half_precision = True`` in the model's config
        (a script sets it at top level), runs the forward under autocast:
        float16 on a GPU (0.28 s instead of 0.61 s for fly_organelles_run08
        on a 2080 Ti), bfloat16 on a CPU. It is kept only if, at warmup, its
        output is within HALF_PRECISION_TOLERANCE of fp32's; otherwise a
        warning says by how much it differed and the model is served in fp32.
        Unlike ``use_half_prediction``, which converts the weights, the
        weights stay in fp32 and torch picks the operations that are safe in
        half precision. A config's own predict or process_chunk decides its
        precision itself.
        """
        import torch

        self.device_slots = device_slots

        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")
            logger.warning("No GPU available, using CPU")
        # torch.backends.cudnn.allow_tf32 = True  # May help performance with newer cuDNN
        # torch.backends.cudnn.enabled = True
        # torch.backends.cudnn.benchmark = True  # Find best algorithm for the hardware

        self.use_half_prediction = use_half_prediction
        self.model_config = model_config
        # Populated by the warmup probe; None when it could not run.
        self.output_range = None
        self.output_class = None
        # The warmup forward below checks the declared shapes, on the device
        # that serves; building the config then needs no forward of its own.
        model_config.check_shapes_on_warmup = True
        # config is lazy so one call is needed to get the config
        _ = self.model_config.config
        self.half_precision = bool(
            half_precision or getattr(self.model_config.config, "half_precision", False)
        )
        # The dtype chunks are computed in under autocast; set by the warmup
        # when half precision is asked for and agrees with fp32.
        self.autocast_dtype = None

        if hasattr(self.model_config.config, "read_shape") and hasattr(
            self.model_config.config, "write_shape"
        ):
            self.context = (
                Coordinate(self.model_config.config.read_shape)
                - Coordinate(self.model_config.config.write_shape)
            ) / 2

        self.optimize_model()
        if not hasattr(self.model_config.config, "predict"):
            logger.warning("No predict function provided, using default")
            self.model_config.config.predict = predict

    def optimize_model(self):
        import torch

        if not hasattr(self.model_config.config, "model"):
            logger.error("Model is not loaded, cannot optimize")
            return
        if not isinstance(self.model_config.config.model, torch.nn.Module):
            logger.warning("Model is not a nn.Module, we only optimize torch models")
            return
        self.model_config.config.model.to(self.device)
        if self.use_half_prediction:
            self.model_config.config.model.half()
        print(f"Using device: {self.device}")
        # DIDN'T WORK with unet model
        # if torch.__version__ >= "2.0":
        #     self.model_config.config.model = torch.compile(self.model_config.config.model)
        # print("Model compiled")
        self.model_config.config.model.eval()
        self._warmup()

    def _warmup(self):
        """Run one throwaway forward pass so the first real chunk doesn't pay for it.

        ``.to(device)`` only moves weights; cuDNN algorithm selection, CUDA
        kernel module loading and workspace allocation all happen lazily on the
        first actual ``forward()``. Without this, that one-time cost lands on
        whichever chunk neuroglancer happens to request first, which presents as
        "the layer appeared but nothing loads for a while". Doing it here moves
        the stall to server startup, where the dashboard is already blocked in
        ``wait_for_host()`` and it costs the user nothing.

        Best effort: a failure here (unknown shapes, an unusual forward
        signature, OOM) must never stop the server from coming up. The one
        exception is an output whose shape contradicts the config's declared
        write_shape, block_shape or output_channels: that raises, as building
        the config used to, since every chunk served would be misplaced.
        """
        import torch

        config = self.model_config.config
        try:
            input_size = getattr(config, "input_size", None)
            if input_size is None:
                # read_shape is in world units; convert to voxels the same way
                # ModelConfig does.
                input_size = np.array(config.read_shape) // np.array(
                    config.input_voxel_size
                )
            shape = (1, 1, *(int(s) for s in input_size))
        except Exception as e:
            logger.info(f"Skipping warmup, could not determine input shape: {e}")
            return

        try:
            # Deliberately extreme inputs rather than zeros: this same pass
            # doubles as the output-activation probe below, and only inputs far
            # outside the trained range reveal whether the model saturates.
            dummy = torch.randn(shape, device=self.device) * 100
            dummy = dummy.half() if self.use_half_prediction else dummy.float()
            start = time.time()
            with torch.no_grad():
                out = config.model.forward(dummy)
            if self.device.type == "cuda":
                torch.cuda.synchronize()
            logger.info(f"Warmup forward {shape} took {time.time() - start:.1f}s")
        except Exception as e:
            logger.warning(
                f"Warmup forward {shape} failed ({e}); the first chunk request "
                "will absorb the one-time initialization cost instead"
            )
            return
        check = getattr(self.model_config, "check_output_shape", None)
        if check is not None and getattr(self.model_config, "validate_model_shapes", True):
            check(out.shape)
        self._record_output_class(out)
        if self.half_precision:
            self._check_half_precision(shape)

    def _check_half_precision(self, shape):
        """Turn autocast on if it agrees with fp32 on the model's output.

        The comparison input is uniform in [-1, 1], the range most input
        chains produce, rather than the extreme warmup input above, which
        can overflow float16 inside the network in a way real data doesn't.
        These forwards also initialize the half-precision kernels, so the
        first chunk doesn't pay for that.
        """
        import torch

        config = self.model_config.config
        name = getattr(self.model_config, "name", None) or type(self.model_config).__name__
        if self.use_half_prediction:
            return  # the weights are already half precision
        own_predict = getattr(config, "predict", predict) is not predict
        if own_predict or callable(getattr(config, "process_chunk", None)):
            logger.warning(
                f"Half precision asked for {name}, but its config has its own "
                "predict or process_chunk, which decides its precision; not used"
            )
            return
        dtype = torch.float16 if self.device.type == "cuda" else torch.bfloat16
        try:
            probe = torch.rand(shape, device=self.device) * 2 - 1
            with torch.no_grad():
                reference = config.model.forward(probe).float()
                with _autocast(self.device, dtype):
                    # Twice: TorchScript re-optimizes a graph on its second
                    # call with new input types, which cost the first chunk
                    # 0.4 s (fly_organelles_run08, 2080 Ti).
                    config.model.forward(probe)
                    reduced = config.model.forward(probe).float()
            difference = float((reduced - reference).abs().max())
            spread = float(reference.max() - reference.min()) or 1.0
        except Exception as e:
            logger.warning(f"{name}: {dtype} forward failed ({e}); serving in fp32")
            return
        relative = difference / spread
        if not np.isfinite(relative) or relative > HALF_PRECISION_TOLERANCE:
            logger.warning(
                f"{name}: {dtype} output differs from fp32 by up to {difference:.3g} "
                f"({relative:.1%} of the output's range, over the "
                f"{HALF_PRECISION_TOLERANCE:.0%} allowed); serving in fp32"
            )
            return
        self.autocast_dtype = dtype
        logger.info(
            f"{name}: serving in {dtype} under autocast; it differs from fp32 by "
            f"up to {difference:.3g} ({relative:.2%} of the output's range)"
        )

    def _record_output_class(self, out):
        """Classify the model's output activation from the warmup pass.

        Stored on the instance so the server can report it to the dashboard,
        which uses it to suggest (or sanity-check) the postprocessing chain.
        """
        from cellmap_flow.utils.output_probe import classify_output_range

        try:
            lo, hi = float(out.min()), float(out.max())
            if not (np.isfinite(lo) and np.isfinite(hi)):
                logger.info("Output probe saw non-finite values; skipping")
                return
            self.output_range = (lo, hi)
            self.output_class = classify_output_range(lo, hi)
            logger.info(
                f"Model output range [{lo:.4g}, {hi:.4g}] -> {self.output_class}"
            )
        except Exception as e:
            logger.info(f"Could not classify model output: {e}")

    def predict(self, idi, out_roi, cancelled=None):
        """The model's raw output for ``out_roi``: ``(C, *spatial)``, not postprocessed.

        ``idi`` is read around ``out_roi`` with the model's context, normalized
        however ``idi`` normalizes. ``cancelled``: asked while the chunk waits
        for a device slot; raises ChunkCancelled, without computing it, once
        that says yes.
        """
        own = getattr(self.model_config.config, "process_chunk", None)
        if own and callable(own):
            # A config's own process_chunk (TF, ONNX, cellpose, bioimage) runs
            # its model somewhere inside, so all of it takes the slot.
            with _device_part(self.device_slots, cancelled):
                return own(idi, out_roi)
        return self.process_chunk_basic(idi, out_roi, cancelled)

    def process_chunk_basic(self, idi, roi, cancelled=None):
        output_roi = roi

        input_roi = output_roi.grow(self.context, self.context)
        kwargs = dict(
            idi=idi, device=self.device, use_half_prediction=self.use_half_prediction
        )
        if self.model_config.config.predict is predict:
            # The default predict takes the slot around its device part only.
            return predict(
                input_roi,
                output_roi,
                self.model_config.config,
                device_slots=self.device_slots,
                cancelled=cancelled,
                autocast_dtype=self.autocast_dtype,
                **kwargs,
            )
        # A script's own predict may not accept more keywords, and its device
        # part can't be told apart from the rest, so all of it takes the slot.
        with _device_part(self.device_slots, cancelled):
            return self.model_config.config.predict(
                input_roi, output_roi, self.model_config.config, **kwargs
            )
