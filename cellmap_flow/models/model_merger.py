"""
Model Merger classes for combining outputs from multiple models.

Similar to PostProcessor pattern, provides subclasses for different merge strategies
that can be looked up by their __name__ attribute.
"""

import logging
import numpy as np
import inspect

logger = logging.getLogger(__name__)


class ModelMerger:
    """Base class for model merging strategies."""

    def merge(self, model_outputs):
        """
        Merge outputs from multiple models.

        Args:
            model_outputs: List of numpy arrays from different models

        Returns:
            Merged numpy array
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} must implement merge() method"
        )


class AndModelMerger(ModelMerger):
    """
    AND merge strategy: Element-wise minimum.
    For binary data: logical AND. For continuous data: minimum value.
    """

    def merge(self, model_outputs):
        """
        Apply AND operation: element-wise minimum across all model outputs.

        Args:
            model_outputs: List of numpy arrays

        Returns:
            Merged array with minimum values at each position
        """
        merged = model_outputs[0].copy()
        for output in model_outputs[1:]:
            merged = np.minimum(merged, output)
        return merged


class OrModelMerger(ModelMerger):
    """
    OR merge strategy: Element-wise maximum.
    For binary data: logical OR. For continuous data: maximum value.
    """

    def merge(self, model_outputs):
        """
        Apply OR operation: element-wise maximum across all model outputs.

        Args:
            model_outputs: List of numpy arrays

        Returns:
            Merged array with maximum values at each position
        """
        merged = model_outputs[0].copy()
        for output in model_outputs[1:]:
            merged = np.maximum(merged, output)
        return merged


class SumModelMerger(ModelMerger):
    """
    SUM merge strategy: Average of all outputs.
    Computes the mean across all model outputs.
    """

    def merge(self, model_outputs):
        """
        Apply SUM operation: average all model outputs.

        Args:
            model_outputs: List of numpy arrays

        Returns:
            Merged array with average values at each position
        """
        merged = np.sum(model_outputs, axis=0) / len(model_outputs)
        return merged


def get_model_mergers_list() -> list[dict]:
    """
    Returns a list of dictionaries containing information about all ModelMerger subclasses.

    Returns:
        List of dicts with keys: 'class_name', 'name', 'description'
    """
    merger_classes = ModelMerger.__subclasses__()
    mergers = []
    for merger_cls in merger_classes:
        merger_name = merger_cls.__name__
        # Extract description from docstring
        description = (
            merger_cls.__doc__.strip().split("\n")[0]
            if merger_cls.__doc__
            else "No description available"
        )
        mergers.append(
            {
                "class_name": merger_name,
                "name": merger_name.replace("ModelMerger", ""),  # e.g., "And"
                "description": description,
            }
        )
    return mergers


def get_model_merger(merger_name: str) -> ModelMerger:
    """
    Get a ModelMerger instance by name.

    Args:
        merger_name: Name of the merger class (e.g., 'AndModelMerger', 'And', 'AND')

    Returns:
        Instance of the requested ModelMerger subclass

    Raises:
        ValueError: If merger_name doesn't match any available merger class
    """
    merger_classes = ModelMerger.__subclasses__()

    # Normalize the input name for comparison
    normalized_name = merger_name.upper().strip()

    for merger_cls in merger_classes:
        class_name = merger_cls.__name__
        short_name = class_name.replace("ModelMerger", "").upper()

        if normalized_name in [class_name.upper(), short_name]:
            return merger_cls()

    available = [cls.__name__ for cls in merger_classes]
    raise ValueError(
        f"Unknown merger: {merger_name}. Available mergers: {available}"
    )


# Convenience mapping for common merge mode strings
MERGE_MODE_MAP = {
    "AND": "AndModelMerger",
    "OR": "OrModelMerger",
    "SUM": "SumModelMerger",
}
