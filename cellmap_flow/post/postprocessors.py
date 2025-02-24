import logging
import numpy as np
import inspect
from cellmap_flow.utils.web_utils import encode_to_str, decode_to_json

logger = logging.getLogger(__name__)


class PostProcessor:

    @classmethod
    def name(cls):
        return cls.__name__

    def __call__(self, data: np.ndarray) -> np.ndarray:
        return self.process(data)

    def process(self, data) -> np.ndarray:
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

class DefaultPostprocessor(PostProcessor):
    def __init__(self):
        pass

    def _process(self, data):
        data = data.clip(-1, 1)
        data = (data + 1) * 255.0 / 2.0
        return data.astype(np.uint8) 
    
    def to_dict(self):
        return {"name": self.name()}
    
    @property
    def dtype(self):
        return np.uint8



class LambdaPostprocessor(PostProcessor):
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


def get_postprocessors_list()-> list[dict]:
    """Returns a list of dictionaries containing the names and parameters of all subclasses of PostProcessor."""
    postprocess_classes = PostProcessor.__subclasses__()
    postoricessors = []
    for post_cls in postprocess_classes:
        post_name = post_cls.__name__
        sig = inspect.signature(post_cls.__init__)
        params = {}
        for param_name, param_obj in sig.parameters.items():
            if param_name == "self":
                continue
            default_val = param_obj.default
            if default_val is inspect._empty:
                default_val = ""
            params[param_name] = default_val
        postoricessors.append(
            {
                "class_name": post_cls.__name__,
                "name": post_name,
                "params": params,
            }
        )
    return postoricessors


def get_postprocessors(elms: dict) -> PostProcessor:
    result = []
    for post_name in elms:
        found = False
        for nm in PostProcessorMethods:
            if nm.name() == post_name:
                result.append(nm(**elms[post_name]))
                found = True
                break
        if not found:
            raise ValueError(f"PostProcess method {post_name} not found")
    return result



PostProcessorMethods = [f for f in PostProcessor.__subclasses__()]