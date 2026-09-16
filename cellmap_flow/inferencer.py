# %%
import time
import numpy as np
import torch
from funlib.geometry import Coordinate
import logging
from cellmap_flow.models.models_config import ModelConfig
from cellmap_flow.globals import g


logger = logging.getLogger(__name__)


def apply_postprocess(data, **kwargs):

    for pross in g.postprocess:
        # logger.error(f"applying postprocess: {pross}")
        data = pross(data, **kwargs)
    return data


def predict(read_roi, write_roi, config, **kwargs):
    idi = kwargs.get("idi")
    if idi is None:
        raise ValueError("idi must be provided in kwargs")

    device = kwargs.get("device")
    if device is None:
        raise ValueError("device must be provided in kwargs")

    use_half_prediction = kwargs.get("use_half_prediction", False)

    raw_input = idi.to_ndarray_ts(read_roi)
    raw_input = np.expand_dims(raw_input, (0, 1))

    with torch.no_grad():
        raw_input_torch = torch.from_numpy(raw_input).to(device, non_blocking=True)
        logger.error(f"Predicting with model {type(config.model).__name__} on device {device}")
        logger.error(f"Input shape: {raw_input_torch.shape}, dtype: {raw_input_torch.dtype}")
        raw_input_torch = raw_input_torch.half() if use_half_prediction else raw_input_torch.float()
        result = config.model.forward(raw_input_torch).cpu().numpy()[0]
        logger.error(f"Output shape: {result.shape}, dtype: {result.dtype}")
    return result

class Inferencer:
    def __init__(self, model_config: ModelConfig, use_half_prediction=False):

        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")
            logger.error("No GPU available, using CPU")
        # torch.backends.cudnn.allow_tf32 = True  # May help performance with newer cuDNN
        # torch.backends.cudnn.enabled = True
        # torch.backends.cudnn.benchmark = True  # Find best algorithm for the hardware

        self.use_half_prediction = use_half_prediction
        self.model_config = model_config
        # Populated by the warmup probe; None when it could not run.
        self.output_range = None
        self.output_class = None
        # config is lazy so one call is needed to get the config
        _ = self.model_config.config

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
        if not hasattr(self.model_config.config, "model"):
            logger.error("Model is not loaded, cannot optimize")
            return
        if not isinstance(self.model_config.config.model, torch.nn.Module):
            logger.error("Model is not a nn.Module, we only optimize torch models")
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
        signature, OOM) must never stop the server from coming up.
        """
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
            self._record_output_class(out)
        except Exception as e:
            logger.warning(
                f"Warmup forward {shape} failed ({e}); the first chunk request "
                "will absorb the one-time initialization cost instead"
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

    def process_chunk(self, idi, roi):
        # check if process_chunk is in self.config
        if getattr(self.model_config.config, "process_chunk", None) and callable(
            self.model_config.config.process_chunk
        ):
            result = self.model_config.config.process_chunk(idi, roi)
        else:
            result = self.process_chunk_basic(idi, roi)

        postprocessed = apply_postprocess(
            result,
            chunk_corner=tuple(roi.get_begin() // roi.get_shape()),
            chunk_num_voxels=np.prod(roi.get_shape() // idi.output_voxel_size),
        )
        return postprocessed

    def process_chunk_basic(self, idi, roi):
        output_roi = roi

        input_roi = output_roi.grow(self.context, self.context)
        result = self.model_config.config.predict(
            input_roi,
            output_roi,
            self.model_config.config,
            idi=idi,
            device=self.device,
            use_half_prediction=self.use_half_prediction,
        )
        return result
