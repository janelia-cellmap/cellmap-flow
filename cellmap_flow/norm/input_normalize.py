import logging
import numpy as np
import inspect
from skimage.morphology import dilation, cube
from edt import edt

logger = logging.getLogger(__name__)


class SerializableInterface:

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
        return data.astype(self.dtype)

    def _process(self, data):
        raise NotImplementedError("Subclasses must implement this method")

    def to_dict(self):
        result = {}
        result = {"name": self.name()}
        for k, v in self.__dict__.items():
            if not k.startswith("_"):
                result[k] = v
        return result
        # return {self.name():result}

    @property
    def dtype(self):
        return None


class InputNormalizer(SerializableInterface):
    pass


class Dilate(InputNormalizer):
    def __init__(self, size=1):
        self.size = int(size)

    def _process(self, data) -> np.ndarray:
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
        self.anisotropy = tuple((int(anisotropy), int(anisotropy), int(anisotropy)))
        if type == "edt":
            self._func = edt.edt
        elif type == "sdf":
            self._func = edt.sdf
        else:
            raise ValueError("type must be either 'edt' or 'sdf'")
        self.black_border = bool(black_border)
        self.parallel = int(parallel)
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
        from edt import edt, sdf

        if not isinstance(data, np.ndarray):
            raise TypeError("Input data must be a numpy array.")

        # Ensure the data is in uint8 format for distance transform
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

    # Removed redundant dtype property definition.

    def _process(self, data: np.ndarray, **kwargs) -> np.ndarray:

        if not isinstance(data, np.ndarray):
            raise TypeError("Input data must be a numpy array.")

        return edt(
            data.astype(np.uint8),
            anisotropy=self.anisotropy,
            black_border=True,
            parallel=5,
        )


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
        self._lambda = eval(f"lambda x: {expression}")

    def _process(self, data) -> np.ndarray:
        return self._lambda(data.astype(np.float32))

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

    def normalize(self, data: np.ndarray) -> np.ndarray:
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


def deserialize_list(elms: dict, T: type) -> list:
    methods = [f for f in T.__subclasses__()]
    result = []
    for elm_name in elms:
        found = False
        for nm in methods:
            if nm.name() == elm_name:
                result.append(nm(**elms[elm_name]))
                found = True
                break
        if not found:
            raise ValueError(f"method {elm_name} not found")
    return result


def get_normalizations(elms: dict) -> list[InputNormalizer]:
    return deserialize_list(elms, InputNormalizer)
