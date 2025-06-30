Custom Model
========================================

This guide explains how to run your own custom model using the CellMap Flow CLI via the `script` interface.

Overview
--------

To run a custom model:

1. Create a Python script that:
   - Loads and prepares your model.
   - Optionally defines how to process each chunk (advanced).
2. Use the CLI to launch the pipeline.

The CLI command looks like:

.. code-block:: bash

    cellmap_flow script -s /path/to/your_script.py -d /path/to/input_data.zarr -q gpu_h100 -P cellmap

Minimum Requirements for Custom Scripts
---------------------------------------

When using `cellmap_flow script`, your Python script **must define** a configuration object (typically the global scope) that satisfies the following:

Required Attributes
^^^^^^^^^^^^^^^^^^^

Your script must define the following **global-level** variables or attributes:

- **model** *(optional)*: PyTorch model instance. Required if predict is not defined.
- **predict** *(optional)*: Custom callable to run predictions. Required if model is not defined.
- **read_shape** *(required)*: The shape of the input block (in world units or voxels).
- **write_shape** *(required)*: The shape of the output block (i.e., prediction size).
- **input_voxel_size** *(required)*: Size of a voxel in the input data.
- **output_voxel_size** *(required)*: Size of a voxel in the model output.
- **output_channels** *(required)*: Number of channels in the output prediction.
- **block_shape** *(required)*: Shape of each output block (must match write_shape + channels).

Testing Your Script
-------------------

You can test your script locally using:

.. code-block:: bash

    cellmap_flow script-server-check -s /path/to/your_script.py -d /path/to/input.zarr

This will simulate a small 2x2x2 chunk to ensure your setup works correctly.



Basic Script Template (PyTorch)
-------------------------------

If you're using PyTorch and your model is compatible with direct inference via `.forward()` or `.eval()`, you do **not** need to define a `process_chunk` function.

.. code-block:: python

    # pip install fly-organelles
    from fly_organelles.model import StandardUnet
    from funlib.geometry import Coordinate
    import torch
    import numpy as np

    # Voxel size and chunk shape
    input_voxel_size = (8, 8, 8)
    output_voxel_size = Coordinate((8, 8, 8))
    read_shape = Coordinate((178, 178, 178)) * Coordinate(input_voxel_size)
    write_shape = Coordinate((56, 56, 56)) * Coordinate(input_voxel_size)

    def load_eval_model(num_labels, checkpoint_path):
        model_backbone = StandardUnet(num_labels)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        checkpoint = torch.load(checkpoint_path, weights_only=True, map_location=device)
        model_backbone.load_state_dict(checkpoint["model_state_dict"])
        model = torch.nn.Sequential(model_backbone, torch.nn.Sigmoid())
        model.to(device)
        model.eval()
        return model

    CHECKPOINT_PATH = "/path/to/your/model_checkpoint"
    output_channels = 8  # Set according to your model
    model = load_eval_model(output_channels, CHECKPOINT_PATH)
    block_shape = np.array((56, 56, 56, output_channels))

.. note::
   You **must define** the `model` and `block_shape` variables in the global scope. CellMap Flow will use these automatically.

Advanced Usage: Custom Process Function (TensorFlow)
----------------------------------------------------

For advanced use cases or non-standard frameworks like TensorFlow v1, define a `process_chunk` function that handles:

- Rescaling input
- Feeding to model
- Retrieving and postprocessing output

.. code-block:: python

    import tensorflow.compat.v1 as tf
    import os, json
    import numpy as np
    from funlib.geometry import Coordinate, Roi
    from cellmap_flow.image_data_interface import ImageDataInterface

    # Define voxel sizes and context
    voxel_size = Coordinate((8, 8, 8))
    output_voxel_size = Coordinate((8, 8, 8))
    read_shape = Coordinate((268, 268, 268)) * voxel_size
    write_shape = Coordinate((164, 164, 164)) * output_voxel_size
    context = (read_shape - write_shape) / 2

    output_channels = 10
    block_shape = np.array((164, 164, 164, output_channels))

    # Load TensorFlow model
    def load_eval_model(setup_dir, checkpoint):
        graph = tf.Graph()
        session = tf.Session(graph=graph)
        with graph.as_default():
            meta_graph_file = os.path.join(setup_dir, "config.meta")
            saver = tf.train.import_meta_graph(meta_graph_file)
            saver.restore(session, os.path.join(setup_dir, checkpoint))
        return session

    setup_dir = "/path/to/tf_model_dir"
    checkpoint = "train_net_checkpoint_400000"
    session = load_eval_model(setup_dir, checkpoint)

    def get_tensor_names(setup_dir, inputs, outputs):
        with open(os.path.join(setup_dir, "config.json"), "r") as f:
            net_config = json.load(f)
        return [net_config[it] for it in inputs], [net_config[ot] for ot in outputs]

    def rescale_data(input_array, min_val, max_val):
        return (2.0 * (input_array - min_val) / (max_val - min_val)) - 1.0

    def process_lsd(chunk, session, input_tensorname, output_tensorname):
        input_data = rescale_data(chunk, 158, 233)
        result = session.run(
            {ot: ot for ot in output_tensorname},
            feed_dict={input_tensorname[0]: input_data}
        )
        return (result[output_tensorname[0]].clip(0, 1) * 255).astype(np.uint8)

    def process_chunk(idi: ImageDataInterface, input_roi: Roi):
        input_roi = input_roi.grow(context, context)
        chunk = idi.to_ndarray_ts(input_roi)
        input_tensor, output_tensor = get_tensor_names(setup_dir, ["raw"], ["embedding"])
        return process_lsd(chunk, session, input_tensor, output_tensor)



