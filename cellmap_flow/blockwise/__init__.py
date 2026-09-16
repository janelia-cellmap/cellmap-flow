"""Blockwise processing.

``CellMapFlowBlockwiseProcessor`` is re-exported lazily: importing it pulls in
the whole inference stack, and ``cellmap_flow_blockwise --help`` -- which goes
through this package to reach cli.py -- paid ~18s for it before printing
anything.
"""

__all__ = ["CellMapFlowBlockwiseProcessor"]


def __getattr__(name):
    if name in __all__:
        from cellmap_flow.blockwise.blockwise_processor import (
            CellMapFlowBlockwiseProcessor,
        )

        return CellMapFlowBlockwiseProcessor
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
