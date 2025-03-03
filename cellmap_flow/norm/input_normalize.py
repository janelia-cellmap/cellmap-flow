import logging
import numpy as np
import inspect

logger = logging.getLogger(__name__)


class InputNormalizer:

    @classmethod
    def name(cls):
        return cls.__name__

    def __call__(self, data: np.ndarray) -> np.ndarray:
        return self.normalize(data)

    def normalize(self, data) -> np.ndarray:
        if not isinstance(data, np.ndarray):
            data = np.array(data)
        if data.dtype.kind in {"U", "O"}:
            try:
                data = data.astype(self.dtype)
            except ValueError:
                raise TypeError(
                    f"Cannot convert non-numeric data to float. Found dtype: {data.dtype}"
                )

        data = self._process(data)
        return data.astype(self.dtype)

    def _process(self, data):
        raise NotImplementedError("Subclasses must implement this method")

    def to_dict(self):
        result = {"name": self.name()}
        for k, v in self.__dict__.items():
            result[k] = v
        return result

    @property
    def dtype(self):
        return np.uint8


class MinMaxNormalizer(InputNormalizer):
    def __init__(self, min_value=0.0, max_value=255.0):
        self.min_value = float(min_value)
        self.max_value = float(max_value)

    @property
    def dtype(self):
        return np.float32

    def _process(self, data) -> np.ndarray:
        data = data.clip(self.min_value, self.max_value)
        return (data - self.min_value) / (self.max_value - self.min_value)


class LambdaNormalizer(InputNormalizer):
    def __init__(self, expression: str):
        self.expression = expression
        self._lambda = eval(f"lambda x: {expression}")

    def _process(self, data) -> np.ndarray:
        return self._lambda(data.astype(np.float32))

    def to_dict(self):
        return {"name": self.name(), "expression": self.expression}

    @property
    def dtype(self):
        return np.float32


class ZScoreNormalizer(InputNormalizer):

    def __init__(self, mean=0.0, std=1.0):
        self.mean = mean
        self.std = std

    @property
    def dtype(self):
        return np.float32

    def normalize(self, data: np.ndarray) -> np.ndarray:
        return (data - self.mean) / self.std


NormalizationMethods = [f for f in InputNormalizer.__subclasses__()]


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


def get_normalizations(elms: dict) -> InputNormalizer:
    result = []
    for norm_name in elms:
        found = False
        for nm in NormalizationMethods:
            if nm.name() == norm_name:
                result.append(nm(**elms[norm_name]))
                found = True
                break
        if not found:
            raise ValueError(f"Normalization method {norm_name} not found")
    return result
