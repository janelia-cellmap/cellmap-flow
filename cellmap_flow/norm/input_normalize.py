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
    

class MinMaxNormalizer(InputNormalizer):

    @classmethod
    def name(cls):
        return "min_max"

    def __init__(self, min_value = 0.0, max_value = 255.0):
        self.min_value = min_value
        self.max_value = max_value

    def normalize(self, data: np.ndarray) -> np.ndarray:
        data = data.astype(np.float32)
        data.clip(self.min_value, self.max_value)
        return ((data - self.min_value) / (self.max_value - self.min_value)).astype(np.float32) 