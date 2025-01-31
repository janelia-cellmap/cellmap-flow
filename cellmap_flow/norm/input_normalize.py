import logging
import numpy as np
import inspect

logger = logging.getLogger(__name__)


class InputNormalizer:

    @classmethod
    def name(cls):
        return cls.__name__

    def normalize(self, data):
        logger.error("InputNormalizer.normalize not implemented")
        return data

    def to_dict(self):
        result = {"name": self.name()}
        for k, v in self.__dict__.items():
            result[k] = v
        return result


class MinMaxNormalizer(InputNormalizer):

    def __init__(self, min_value=0.0, max_value=255.0):
        self.min_value = min_value
        self.max_value = max_value

    def normalize(self, data: np.ndarray) -> np.ndarray:
        data = data.astype(np.float32)
        data = data.clip(self.min_value, self.max_value)
        return ((data - self.min_value) / (self.max_value - self.min_value)).astype(
            np.float32
        )


NormalizationMethods = [f.name() for f in InputNormalizer.__subclasses__()]


def get_normalizers():
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


def get_normalization(elms: dict) -> InputNormalizer:
    if "name" not in elms:
        raise ValueError(f"Normalization method name not specified in {elms}")
    name = elms["name"]
    rest_elm = elms.copy()
    rest_elm.pop("name")
    for nm in NormalizationMethods:
        if nm.name() == name:
            return nm(**rest_elm)
    raise ValueError(f"Normalization method {name} not found")
