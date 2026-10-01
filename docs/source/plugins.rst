Plugins
=======

CellMap Flow supports user-defined **plugins** — custom normalizers, postprocessors, and model configurations that you can register once and use everywhere (CLI, dashboard, YAML pipelines).

Plugins are Python files stored in ``~/.cellmap_flow/plugins/`` and loaded every time a ``cellmap_flow`` command or the dashboard starts. Importing ``cellmap_flow`` does not load them: a Python script that uses a plugin's classes calls ``cellmap_flow.plugins.load_plugins()`` first.

CLI Commands
------------

Register a plugin
~~~~~~~~~~~~~~~~~~

.. code-block:: bash

    cellmap_flow plugins register /path/to/my_plugin.py

This copies the file to ``~/.cellmap_flow/plugins/my_plugin.py``. A basic safety check is performed to reject scripts that import ``os``, ``subprocess``, ``sys`` or call ``eval``/``exec``.

To overwrite an existing plugin with the same filename:

.. code-block:: bash

    cellmap_flow plugins register /path/to/my_plugin.py --force

List registered plugins
~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

    cellmap_flow plugins list

Output:

.. code-block:: text

    Registered plugins:

      my_normalizer.py  (/home/user/.cellmap_flow/plugins/my_normalizer.py)
      my_post.py        (/home/user/.cellmap_flow/plugins/my_post.py)

Unregister a plugin
~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

    cellmap_flow plugins unregister my_normalizer

The ``.py`` extension is optional. This deletes the file from the plugins directory.

Writing Plugins
---------------

A plugin is a plain Python file that defines one or more subclasses of
``InputNormalizer``, ``PostProcessor``, or ``ModelConfig``. Each subclass must:

1. Inherit from the correct base class.
2. Accept all ``__init__`` parameters as simple types (``float``, ``int``, ``str``, ``bool``) so they can be set from the dashboard or YAML.
3. Implement the required abstract methods.

Custom Normalizer
~~~~~~~~~~~~~~~~~

Subclass ``InputNormalizer`` and implement ``_process`` and ``dtype``.

.. code-block:: python

    from cellmap_flow.norm.input_normalize import InputNormalizer
    import numpy as np


    class MyNewNormalizer(InputNormalizer):
        """Scale input data by a constant coefficient."""

        def __init__(self, coef: float = 1.0):
            self.coef = float(coef)

        @property
        def dtype(self):
            return np.float32

        def _process(self, data: np.ndarray) -> np.ndarray:
            return data * self.coef

Another example — percentile-based normalization:

.. code-block:: python

    from cellmap_flow.norm.input_normalize import InputNormalizer
    import numpy as np


    class PercentileNormalizer(InputNormalizer):
        """Normalize data to a given percentile range."""

        def __init__(self, low: float = 1.0, high: float = 99.0):
            self.low = float(low)
            self.high = float(high)

        @property
        def dtype(self):
            return np.float32

        def _process(self, data: np.ndarray) -> np.ndarray:
            p_low = np.percentile(data, self.low)
            p_high = np.percentile(data, self.high)
            if p_high - p_low == 0:
                return np.zeros_like(data, dtype=np.float32)
            return ((data - p_low) / (p_high - p_low)).clip(0, 1).astype(np.float32)

Custom PostProcessor
~~~~~~~~~~~~~~~~~~~~

Subclass ``PostProcessor`` and implement ``_process``, ``dtype``, and ``is_segmentation``.

.. code-block:: python

    from cellmap_flow.post.postprocessors import PostProcessor
    import numpy as np


    class SigmoidPostprocessor(PostProcessor):
        """Apply sigmoid activation and convert to uint8."""

        def __init__(self, scale: float = 1.0):
            self.scale = float(scale)

        def _process(self, data: np.ndarray) -> np.ndarray:
            result = 1.0 / (1.0 + np.exp(-data * self.scale))
            return (result * 255).astype(np.uint8)

        @property
        def dtype(self):
            return np.uint8

        @property
        def is_segmentation(self):
            return False

Another example — connected-component filtering by size:

.. code-block:: python

    from cellmap_flow.post.postprocessors import PostProcessor
    import numpy as np
    from scipy.ndimage import label


    class SizeFilterPostprocessor(PostProcessor):
        """Remove connected components smaller than min_size voxels."""

        def __init__(self, min_size: int = 100):
            self.min_size = int(min_size)

        def _process(self, data: np.ndarray) -> np.ndarray:
            labeled, num_features = label(data > 0)
            for idx in range(1, num_features + 1):
                if np.sum(labeled == idx) < self.min_size:
                    data[labeled == idx] = 0
            return data

        @property
        def dtype(self):
            return np.uint8

        @property
        def is_segmentation(self):
            return True

Custom ModelConfig
~~~~~~~~~~~~~~~~~~

Subclass ``ModelConfig`` and implement ``_get_config``.

``command`` (how a server is launched for this config) and ``to_dict`` (how it is exported to a YAML) have defaults, built from the constructor arguments the config was created with. Override them only if the config needs something else. The type is named by ``cli_name`` when the class sets one, otherwise by its class name without ``ModelConfig``, lower-cased (``ONNXModelConfig`` is ``onnx``). That name is what ``type:`` in a YAML and the ``cellmap_flow`` subcommand use.

The ``_get_config`` method must return a ``Config`` object with the following attributes:
``model`` (or ``predict``), ``read_shape``, ``write_shape``, ``input_voxel_size``,
``output_voxel_size``, ``output_channels``, and ``block_shape``.

.. code-block:: python

    import numpy as np
    import torch
    from funlib.geometry import Coordinate
    from cellmap_flow.models.models_config import Config, ModelConfig


    class ONNXModelConfig(ModelConfig):
        """Load a model from an ONNX file."""

        def __init__(
            self,
            onnx_path: str,
            input_voxel_size: str = "8,8,8",
            output_voxel_size: str = "8,8,8",
            input_shape: str = "128,128,128",
            output_shape: str = "128,128,128",
            output_channels: int = 1,
            name: str = None,
            scale: str = None,
        ):
            super().__init__()
            self.onnx_path = onnx_path
            self._input_voxel_size = tuple(int(v) for v in input_voxel_size.split(","))
            self._output_voxel_size = tuple(int(v) for v in output_voxel_size.split(","))
            self._input_shape = tuple(int(v) for v in input_shape.split(","))
            self._output_shape = tuple(int(v) for v in output_shape.split(","))
            self._output_channels = int(output_channels)
            self.name = name
            self.scale = scale

        def _get_config(self):
            import onnxruntime as ort

            config = Config()
            config.predict = ort.InferenceSession(self.onnx_path).run
            config.input_voxel_size = Coordinate(self._input_voxel_size)
            config.output_voxel_size = Coordinate(self._output_voxel_size)
            config.read_shape = Coordinate(self._input_shape) * config.input_voxel_size
            config.write_shape = Coordinate(self._output_shape) * config.output_voxel_size
            config.output_channels = self._output_channels
            config.block_shape = np.array(
                self._output_shape + (self._output_channels,)
            )
            return config

Quick Start
-----------

1. Write your plugin (e.g. ``my_normalizer.py``)
2. Register it:

   .. code-block:: bash

       cellmap_flow plugins register my_normalizer.py

3. Open the dashboard — your plugin appears in the normalizers/postprocessors/models list:

   .. code-block:: bash

       cellmap_flow view -d /path/to/data.zarr

4. Or use it in a YAML pipeline:

   .. code-block:: yaml

       input_normalizers:
         - name: MyNewNormalizer
           coef: 2.5

       postprocess:
         - name: SigmoidPostprocessor
           scale: 1.0

Notes
-----

- Plugin filenames must be unique. Use ``--force`` to overwrite.
- Plugins are checked for unsafe imports (``os``, ``subprocess``, ``sys``) and function calls (``eval``, ``exec``) before registration.
- ``__init__`` parameters should use simple types (``float``, ``int``, ``str``, ``bool``) for dashboard and YAML compatibility.
- Registered ``ModelConfig`` subclasses automatically get their own CLI command, ``cellmap_flow infer <type>``.
