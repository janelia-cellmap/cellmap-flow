PostProcessor Module
====================

Overview
--------

The `PostProcessor` module defines a framework for applying various postprocessing operations to `numpy.ndarray` data, primarily used in image segmentation, classification, and data normalization workflows.

Base Class
----------

.. class:: PostProcessor

    Abstract base class for all postprocessing methods.

    .. method:: name()

        Returns the class name.

    .. method:: __call__(data, **kwargs)

        Invokes the `process()` method.

    .. method:: process(data, **kwargs)

        Ensures `data` is a NumPy array and dispatches to `_process`.

    .. method:: _process(data, **kwargs)

        Abstract method to be implemented by subclasses.

    .. method:: to_dict()

        Serializes instance attributes to a dictionary.

    .. attribute:: dtype

        Returns the target NumPy dtype of the processed output.

    .. attribute:: is_segmentation

        Boolean indicating if output represents a segmentation map.

PostProcessor Subclasses
------------------------

DefaultPostprocessor
~~~~~~~~~~~~~~~~~~~~

.. class:: DefaultPostprocessor(clip_min=-1.0, clip_max=1.0, bias=1.0, multiplier=127.5)

Clips, biases, and scales the data. Converts to `uint8`.

- `clip_min`: Minimum clipping value
- `clip_max`: Maximum clipping value
- `bias`: Value added after clipping
- `multiplier`: Scaling factor

ThresholdPostprocessor
~~~~~~~~~~~~~~~~~~~~~~

.. class:: ThresholdPostprocessor(threshold=0.5)

Applies a threshold to binarize data. Produces `uint8` output.

- `threshold`: Threshold for binarization

LabelPostprocessor
~~~~~~~~~~~~~~~~~~

.. class:: LabelPostprocessor(channel=0)

Applies connected-component labeling to a selected channel using `scipy.ndimage.label`.

MortonSegmentationRelabeling
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. class:: MortonSegmentationRelabeling(channel=0)

Relabels segmentation data using Morton-order encoding to ensure chunk-wise uniqueness.

- `channel`: Channel to operate on
- Uses `pymorton.interleave` for chunk-based ID uniqueness

AffinityPostprocessor
~~~~~~~~~~~~~~~~~~~~~

.. class:: AffinityPostprocessor(bias=0.0, neighborhood=...)

Uses multi-scale watershed (`mwatershed`) to generate segmentations from affinity graphs.

- `bias`: Threshold for filtering weak affinities
- `neighborhood`: List of offset vectors for affinity connectivity

SimpleBlockwiseMerger
~~~~~~~~~~~~~~~~~~~~~

.. class:: SimpleBlockwiseMerger(channel=0, face_erosion_iterations=0)

Merges chunk boundaries using edge voxel consistency and `neuroglancer.equivalence_map`.

- `face_erosion_iterations`: Controls erosion on boundary faces before merge
- Maintains a dictionary of voxel ID equivalences between chunks

ChannelSelection
~~~~~~~~~~~~~~~~

.. class:: ChannelSelection(channels="0")

Selects a subset of channels from the input array.

- `channels`: Comma-separated list of channel indices (e.g., `"0,1,2"`)

LambdaPostprocessor
~~~~~~~~~~~~~~~~~~~

.. class:: LambdaPostprocessor(expression)

Applies a user-defined lambda expression to each data point.

- `expression`: Python expression to apply to the data (e.g., `"x * 2"`)
