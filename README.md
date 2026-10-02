**🚧 This repository is still under construction. 🚧**

Please feel free to explore and contribute, but note that there may be frequent changes.



<p align="center">
  <img src="https://raw.githubusercontent.com/janelia-cellmap/cellmap-flow/refs/heads/main/img/CMFLOW_dark.png" alt="CellMapFlow Logo" />
</p>

<p align="center">
  <a href="#"><img src="https://img.shields.io/badge/Status-Under_Construction-orange.svg" alt="Under Construction" /></a>
</p>

<p align="center">
  <strong>Real-time inference is performed using Torch/Tensorflow, Dacapo, and bioimage models on local data or any cloud-hosted data.</strong>
</p>

<p align="center">
  <img src="https://raw.githubusercontent.com/janelia-cellmap/cellmap-flow/refs/heads/main/img/flow.gif" alt="Animated demonstration of CellMapFlow's real-time data processing workflow" />
</p>

<p align="center">
  🚀 Speed up your data processing from months to minutes!
</p>

<p align="center">
  <img src="https://raw.githubusercontent.com/janelia-cellmap/cellmap-flow/refs/heads/main/img/jrc.gif" alt="Real-time data processing visualization" />
</p>





## Installation

To install CellMapFlow, you can use pip:

```bash
pip install cellmap-flow
```

Note that the basic installation does not include DaCapo and BioImage.io core dependencies. To install CellMapFlow with DaCapo support, use the following command:

```bash
pip install cellmap-flow[dacapo]
```

To install CellMapFlow with BioImage.io support, use the following command:

```bash
pip install cellmap-flow[bioimageio]
```

To install CellMapFlow with both DaCapo and BioImage.io support, use the following command:

```bash
pip install cellmap-flow[dacapo,bioimageio]
```

### Installing from GitHub

To get an unreleased branch without cloning:

```bash
pip install "cellmap-flow @ git+https://github.com/janelia-cellmap/cellmap-flow.git"
```

Append `@<branch>` to the URL for a specific branch, and add extras in
brackets after the name, e.g.
`"cellmap-flow[dacapo] @ git+https://github.com/janelia-cellmap/cellmap-flow.git@my-branch"`.

For development, clone and install in editable mode instead:

```bash
git clone https://github.com/janelia-cellmap/cellmap-flow.git
cd cellmap-flow
pip install -e .
```

### Finetuning

Interactive finetuning needs two dependencies that cannot come from PyPI — a
MinIO server (conda-forge only) and a Neuroglancer fork with voxel-annotation
support — so `pip install cellmap-flow[finetune]` on its own is not enough.
See [docs/finetuning.md](docs/finetuning.md) for the full setup.

## Usage

`cellmap_flow` has a subcommand for each job; `cellmap_flow <command> --help`
shows its options, and the [CLI page](docs/source/cli.rst) lists them all,
with the names they had before 0.3.0.

```bash
$ cellmap_flow --help

Commands:
  blockwise  Run the model a blockwise YAML describes over the whole...
  dashboard  Serve the dashboard alone, for a viewer already running.
  doctor     Check the environment: what is installed, what is missing,...
  finetune   The finetune tools.
  infer      Start a model's inference server, then open the viewer on...
  models     List the model types and the arguments each takes.
  plugins    Register, unregister and list plugins.
  serve      Serve one model's predictions, as an inference job does on...
  view       Start CellMap Flow viewer with a dataset.
  yaml       Run multiple model inference jobs from a YAML configuration...

$ cellmap_flow view -d data_path                       # pick models in the dashboard
$ cellmap_flow yaml config.yaml                        # the models a YAML lists
$ cellmap_flow infer dacapo -r my_run -i iteration -d data_path
$ cellmap_flow infer script -s script_path -d data_path
$ cellmap_flow infer bioimage -m model_path -v 8,8,8 -d data_path
```

A data path is a zarr (v2 or v3), N5 or Neuroglancer precomputed volume, on
disk or at an `s3://`, `gs://` or `https://` URL; public buckets are read
anonymously, private ones with your AWS or Google credentials. See
[data paths](docs/source/data_paths.rst).

A model is fed the dataset's level at its own voxel size. When the dataset has
no such level, `--resample` (or `resample: true` in a YAML, or the dashboard's
*Resample* box) resamples the nearest one to it; without it the nearest level
is used as it is, and the dashboard warns that the model sees the wrong scale.

## Using custom script:
This enables using any model by providing a script e.g. [example/model_spec.py](example/model_spec.py)
e.g.
```bash
cellmap_flow infer script -s /groups/cellmap/cellmap/zouinkhim/cellmap-flow/example/model_spec.py -d /nrs/cellmap/data/jrc_mus-cerebellum-1/jrc_mus-cerebellum-1.zarr/recon-1/em/fibsem-uint8/s0 
```

### Script keywords:
Define these variables in your script (`cellmap_flow infer script -s path/to/your_script.py`):
- **model**: 
  The PyTorch model to be used for inference. 
- **input_size**: 
  The voxel shape of the data to be input to the PyTorch model.
- **output_size**: 
  The voxel shape of the data in output by the PyTorch model.
- **input_voxel_size**: 
  The voxel size of the data input to the model.
- **output_voxel_size**: 
  The voxel size of the data output by the model.
- **output_channels**:
  The number of channels in the output of the model.
- **process_chunk** (optional):
  (Optional) A function that takes an ImageDataInterface and an ROI and returns the data to be display. This can be used to run a TensorFlow model or do other custom data processing.

## Using Dacapo model:
which enable inference using a Dacapo model by providing the run name and iteration number
e.g.
```bash
cellmap_flow infer dacapo -r 20241204_finetune_mito_affs_task_datasplit_v3_u21_kidney_mito_default_cache_8_1 -i 700000 -d /nrs/cellmap/data/jrc_ut21-1413-003/jrc_ut21-1413-003.zarr/recon-1/em/fibsem-uint8/s0
```

## Using bioimage-io model:
still in development

## Using TensorFlow model:
To run TensorFlow models, we suggest installing TensorFlow via conda: `conda install tensorflow-gpu==2.16.1`

## Run multiple models at once:
List them in a YAML file ([docs/source/yaml_config.rst](docs/source/yaml_config.rst)) and run
```bash
cellmap_flow yaml config.yaml
```

