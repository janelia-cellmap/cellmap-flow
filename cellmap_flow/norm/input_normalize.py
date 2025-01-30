import logging
import numpy as np

logger = logging.getLogger(__name__)


class InputNormalizer:

    @classmethod
    def name(cls):
        return "basic"

    def normalize(self, data):
        logger.error("InputNormalizer.normalize not implemented")
        return data

    def to_dict(self):
        result = {"name": self.name()}
        for k, v in self.__dict__.items():
            result[k] = v
        return result


class MinMaxNormalizer(InputNormalizer):

    @classmethod
    def name(cls):
        return "min_max"

    def __init__(self, min_value=0.0, max_value=255.0):
        self.min_value = min_value
        self.max_value = max_value

    def normalize(self, data: np.ndarray) -> np.ndarray:
        data = data.astype(np.float32)
        data.clip(self.min_value, self.max_value)
        return ((data - self.min_value) / (self.max_value - self.min_value)).astype(
            np.float32
        )


NormalizationMethods = [f.name() for f in InputNormalizer.__subclasses__()]


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
