import functools
import logging
import numpy as np
import inspect

from cellmap_flow.norm.safe_expression import compile_expression

logger = logging.getLogger(__name__)


def _jsonable(value):
    """Turn numpy scalars/arrays and tuples into plain JSON types.

    Constructor arguments are kept as given, but they end up in a URL blob via
    json.dumps, which rejects numpy types.
    """
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, dict):
        return {k: _jsonable(v) for k, v in value.items()}
    return value


def _record_init_params(init):
    """Wrap ``__init__`` so the instance remembers the arguments it was built with.

    Only the outermost ``__init__`` records: a subclass calling
    ``super().__init__()`` must not overwrite its own arguments with the base
    class's (usually empty) ones.
    """
    sig = inspect.signature(init)

    @functools.wraps(init)
    def wrapper(self, *args, **kwargs):
        if "_init_params" not in self.__dict__:
            params = {}
            try:
                bound = sig.bind(self, *args, **kwargs)
                bound.apply_defaults()
                for pname, value in list(bound.arguments.items())[1:]:
                    kind = sig.parameters[pname].kind
                    if kind is inspect.Parameter.VAR_POSITIONAL:
                        continue
                    if kind is inspect.Parameter.VAR_KEYWORD:
                        params.update(value)
                    else:
                        params[pname] = value
            except TypeError:
                # Let the real __init__ raise its own, clearer error.
                params = None
            self.__dict__["_init_params"] = params
        return init(self, *args, **kwargs)

    wrapper._records_init_params = True
    return wrapper


class SerializableInterface:

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        init = cls.__dict__.get("__init__")
        if init is not None and not getattr(init, "_records_init_params", False):
            cls.__init__ = _record_init_params(init)

    @classmethod
    def name(cls):
        return cls.__name__

    def __call__(self, data: np.ndarray, **kwargs) -> np.ndarray:
        return self.process(data, **kwargs)

    def __str__(self):
        return str(self.to_dict())

    def __repr__(self):
        return str(self.to_dict())

    def process(self, data, **kwargs) -> np.ndarray:
        if not isinstance(data, np.ndarray):
            data = np.array(data)
        if data.dtype.kind in {"U", "O"}:
            try:
                data = data.astype(self.dtype)
            except ValueError:
                raise TypeError(
                    f"Cannot convert non-numeric data to float. Found dtype: {data.dtype}"
                )
        # if there are kwargs
        sig = inspect.signature(self._process)
        [kwargs.pop(k) for k in list(kwargs.keys()) if k not in sig.parameters]
        data = self._process(data, **kwargs)
        if self.dtype is None:
            # No declared dtype means "whatever _process returned";
            # astype(None) would silently promote it to float64.
            return np.asarray(data)
        return np.asarray(data).astype(self.dtype, copy=False)

    def _process(self, data):
        raise NotImplementedError("Subclasses must implement this method")

    def to_dict(self):
        """``{"name": <class name>, **constructor arguments}``.

        Exactly what ``type(self)(**params)`` needs to rebuild this step. The
        public attributes are not that: constructors parse their arguments
        (a neighborhood string becomes a list, "0,1" becomes [0, 1]) and add
        state of their own, so feeding the attributes back in raised TypeError
        or built a different step.

        Each argument is reported with its real type: when the constructor
        stored it under its own name as a plain number, bool or string, that
        stored value is used (0.5, not the "0.5" a form sent); anything it
        parsed into something else keeps the argument as given.
        """
        params = self.__dict__.get("_init_params")
        if params is None:
            # Built without going through __init__ (e.g. unpickled from an
            # older version); the public attributes are the best there is.
            params = {k: v for k, v in self.__dict__.items() if not k.startswith("_")}
        else:
            params = {k: self._stored_or_given(k, v) for k, v in params.items()}
        result = {"name": self.name()}
        result.update({k: _jsonable(v) for k, v in params.items()})
        return result

    def _stored_or_given(self, name, given):
        stored = self.__dict__.get(name, given)
        if isinstance(stored, (bool, int, float, str, np.generic)):
            return stored
        return given

    @property
    def dtype(self):
        return None

    def output_info(self, dtype, channels):
        """``(dtype, channels, is_segmentation)`` of what this step returns.

        ``dtype`` and ``channels`` describe what it is given. By default the
        step's declared ``dtype`` replaces the incoming one (``None`` keeps
        it), a ``num_channels`` attribute (only steps that change the count
        have one) replaces the channel count, and ``is_segmentation`` is the
        step's own, where ``None`` means it does not say. A step whose output
        depends on its input in some other way overrides this.
        """
        own_dtype = self.dtype
        return (
            own_dtype if own_dtype else dtype,
            getattr(self, "num_channels", channels),
            getattr(self, "is_segmentation", None),
        )


class InputNormalizer(SerializableInterface):
    pass


class ChannelSelector(InputNormalizer):
    def __init__(self, channel=0):
        self.channel = int(channel)

    def _process(self, data) -> np.ndarray:
        # No-op: channel selection is applied at the TensorStore level
        return data


class Dilate(InputNormalizer):
    def __init__(self, size=1):
        self.size = int(size)

    def _process(self, data) -> np.ndarray:
        from skimage.morphology import cube, dilation  # ~2s to import

        return dilation(data, cube(self.size))


class EuclideanDistance(InputNormalizer):
    def __init__(
        self,
        anisotropy=50,
        black_border=True,
        parallel=5,
        type="edt",
        activation="tanh",
    ):
        import edt

        if type not in ["edt", "sdf"]:
            raise ValueError("type must be either 'edt' or 'sdf'")
        self.type = type
        self.anisotropy = tuple((int(anisotropy), int(anisotropy), int(anisotropy)))
        if type == "edt":
            self._func = edt.edt
        elif type == "sdf":
            self._func = edt.sdf
        else:
            raise ValueError("type must be either 'edt' or 'sdf'")
        # The dashboard forms send every value as a string, and bool("False")
        # is True.
        if isinstance(black_border, str):
            self.black_border = black_border.strip().lower() == "true"
        else:
            self.black_border = bool(black_border)
        self.parallel = int(parallel)
        if activation in ("", "None", "none"):
            activation = None
        self.activation = (
            lambda x: x
        )  # default to identity if no activation is specified
        if activation is not None:
            if activation == "tanh":
                self.activation = lambda x: np.tanh(x)
            elif activation == "relu":
                self.activation = lambda x: np.maximum(0, x)
            elif activation == "sigmoid":
                self.activation = lambda x: 1 / (1 + np.exp(-x))
            else:
                raise ValueError(
                    "Unsupported activation function: {}".format(activation)
                )

    def _process(self, data):
        if not isinstance(data, np.ndarray):
            raise TypeError("Input data must be a numpy array.")

        result = self._func(
            data,
            anisotropy=self.anisotropy,
            black_border=self.black_border,
            parallel=self.parallel,
        )
        return self.activation(result.astype(np.float32))

    @property
    def dtype(self):
        return np.float32


class MinMaxNormalizer(InputNormalizer):
    def __init__(self, min_value=0.0, max_value=255.0, invert=False):
        self.min_value = float(min_value)
        self.max_value = float(max_value)
        if type(invert) == str:
            self.invert = invert.lower() == "true"
        else:
            self.invert = bool(invert)

    @property
    def dtype(self):
        return np.float32

    def _process(self, data) -> np.ndarray:
        data = data.clip(self.min_value, self.max_value)
        result = (data - self.min_value) / (self.max_value - self.min_value)
        if self.invert:
            result = 1 - result
        return result.astype(np.float32)


class LambdaNormalizer(InputNormalizer):
    def __init__(self, expression: str):
        self.expression = expression
        # Reject anything outside the safe subset now, not on the first chunk.
        compile_expression(expression)
        # The compiled function is not picklable, which breaks multiprocessing
        # workers (e.g. PyTorch DataLoader with spawn). Don't store it on the
        # instance; build it lazily in ``_process`` so it lives only in the
        # worker that needs it. ``__getstate__``/``__setstate__`` further
        # guarantee older pickled instances don't try to round-trip it.

    def _get_lambda(self):
        if not hasattr(self, "_lambda") or self._lambda is None:
            self._lambda = compile_expression(self.expression)
        return self._lambda

    def __getstate__(self):
        state = self.__dict__.copy()
        state.pop("_lambda", None)
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._lambda = None  # rebuilt lazily on first call

    def _process(self, data) -> np.ndarray:
        return self._get_lambda()(data.astype(np.float32))

    @property
    def dtype(self):
        return np.float32


class ZScoreNormalizer(InputNormalizer):

    def __init__(self, mean=0.0, std=1.0):
        self.mean = float(mean)
        self.std = float(std)

    @property
    def dtype(self):
        return np.float32

    def _process(self, data: np.ndarray) -> np.ndarray:
        return (data - self.mean) / self.std


def get_input_normalizers() -> list[dict]:
    normalizer_classes = InputNormalizer.__subclasses__()
    normalizers = []
    for norm_cls in normalizer_classes:
        norm_name = norm_cls.__name__
        sig = inspect.signature(norm_cls.__init__)
        params = {}
        for param_name, param_obj in sig.parameters.items():
            if param_name == "self":
                continue
            default_val = param_obj.default
            if default_val is inspect._empty:
                default_val = ""
            params[param_name] = default_val
        normalizers.append(
            {
                "class_name": norm_cls.__name__,
                "name": norm_name,
                "params": params,
            }
        )
    return normalizers


def deserialize_list(elms, T: type) -> list:
    """
    Deserialize a list of processors from either dict or list format.
    
    Args:
        elms: Either a dict (old format) or list (new ordered format) of processor configs
        T: The base type to deserialize (InputNormalizer or PostProcessor)
    
    Returns:
        List of instantiated processor objects in the specified order
    """
    methods = [f for f in T.__subclasses__()]
    result = []
    
    # Handle new list format (ordered)
    if isinstance(elms, list):
        for elm_config in elms:
            if isinstance(elm_config, dict) and 'name' in elm_config:
                elm_name = elm_config['name']
                params = {k: v for k, v in elm_config.items() if k != 'name'}
                
                found = False
                for nm in methods:
                    if nm.name() == elm_name:
                        result.append(nm(**params))
                        found = True
                        break
                if not found:
                    logger.warning(f"method {elm_name} not found, skipping")
            else:
                logger.warning(f"Invalid element format in list: {elm_config}")
    
    # Handle old dict format (for backward compatibility)
    elif isinstance(elms, dict):
        for elm_name in elms:
            found = False
            for nm in methods:
                if nm.name() == elm_name:
                    result.append(nm(**elms[elm_name]))
                    found = True
                    break
            if not found:
                logger.warning(f"method {elm_name} not found, skipping")
    
    else:
        raise ValueError(f"Expected dict or list, got {type(elms)}")
    
    return result


def get_normalizations(elms) -> list[InputNormalizer]:
    """Get normalizations from either dict or list format."""
    return deserialize_list(elms, InputNormalizer)
