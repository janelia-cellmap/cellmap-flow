from cellmap_flow.norm.input_normalize import get_normalizers, InputNormalizer
import numpy as np
class JeffNormalizer(InputNormalizer):

    def __init__(self, jeff_v=0.0, j=255.0):
        self.jeff_v = jeff_v
        self.j = j

    def normalize(self, data: np.ndarray) -> np.ndarray:
        data = data.astype(np.float32)
        data = data.clip(self.min_value, self.max_value)
        return ((data - self.min_value) / (self.max_value - self.min_value)).astype(
            np.float32
        )
