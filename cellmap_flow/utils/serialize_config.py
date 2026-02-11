import inspect
from typing import Any

from types import ModuleType

DEFAULT_AXES_NAMES = ["x", "y", "z"]

class Config:
    def __init__(self, **kwargs):
        self.axes_names = kwargs.get("axes_names", DEFAULT_AXES_NAMES)
        self.__dict__.update(kwargs)
        self.kwargs = kwargs

    def __str__(self) -> str:
        elms = []
        for k, v in vars(self).items():
            if any(x in k for x in ["kwargs", "__"]) or isinstance(v, ModuleType):
                continue
            if ["model","checkpoint"].__contains__(k):
                elms.append(f"{k}")
                continue
            # if isinstance(v, np.ndarray):
            #     elms.append(f"{k}: type={type(v)} shape={v.shape}\n")
            # elif inspect.ismodule(v):
            #     elms.append(f"{k}: <module '{v.__name__}'>\n")
            # elif k=="checkpoint" or k=="model":
            #     elms.append(f"{k}\n")
            # else:
            elms.append(f"{k}: {v}")
        newline = '\n'
        return f"{type(self).__name__}({newline.join(elms)})"
    
    def __repr__(self) -> str:
        return self.__str__()

    def to_dict(self):
        """
        Returns the configuration as a dictionary.
        """
        return self.kwargs

    def serialize(self):
        """
        Serializes the configuration to a string representation.
        """
        serialized = {}
        for key, value in self.kwargs.items():
            if (
                inspect.ismodule(value)
                or inspect.isclass(value)
                or inspect.isfunction(value)
                or inspect.isbuiltin(value)
            ):
                # Skip modules, classes, and functions
                continue
            elif "__" in key:
                # Skip private attributes
                continue
            elif not isinstance(value, (int, float, str, bool)):
                serialized[key] = str(value)
            else:
                serialized[key] = value
        return serialized

    def get(self, key: str, default: Any = None) -> Any:
        """
        Gets the value of a configuration key.
        """
        return self.kwargs.get(key, default)
