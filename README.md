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

### With pixi (recommended)

[pixi](https://pixi.sh) installs cellmap-flow with everything it needs, and
the separate environments some model families run in:

```bash
git clone https://github.com/janelia-cellmap/cellmap-flow.git
cd cellmap-flow
pixi install                      # the default environment
pixi run cellmap_flow view -d data_path
```

Model families whose dependencies clash with the default environment run in
their own pixi environment, chosen by the model's type (see
[Model types](#model-types)). Each is installed the first time a model needs
it, or ahead of time with `pixi run cellmap_flow envs install <name>`;
`cellmap_flow envs` lists them and what is installed.

### With pip

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
  add        Resolve a model reference into a model entry, and optionally run it.
  envs       The environments models run in, besides this one.
  finetune   The finetune tools.
  infer      Start a model's inference server, then open the viewer on...
  models     List the model types and the arguments each takes.
  plugins    Register, unregister and list plugins.
  serve      Serve one model's predictions, as an inference job does on...
  view       Start CellMap Flow viewer with a dataset.
  yaml       Run multiple model inference jobs from a YAML configuration...

$ cellmap_flow view -d data_path                       # pick models in the dashboard
$ cellmap_flow yaml config.yaml                        # the models a YAML lists
$ cellmap_flow infer huggingface -r cellmap/fly_organelles_run08_438000 -d data_path
$ cellmap_flow infer cellpose -v 64 -d data_path                 # Cellpose-SAM
$ cellmap_flow infer bioimage -m conscientious-dromedary -v 16 -d data_path
$ cellmap_flow infer fly -c /path/to/run/model_checkpoint_20000 -d data_path
$ cellmap_flow infer script -s script_path -d data_path
```

A data path is a zarr (v2 or v3), N5 or Neuroglancer precomputed volume, on
disk or at an `s3://`, `gs://` or `https://` URL; public buckets are read
anonymously, private ones with your AWS or Google credentials. See
[data paths](docs/source/data_paths.rst).

A model is fed the dataset's level at its own voxel size. When the dataset has
no such level, `--resample` (or `resample: true` in a YAML, or the dashboard's
*Resample if no scale matches the model* box) resamples the nearest one to it; without it the nearest level
is used as it is, and the dashboard warns that the model sees the wrong scale.

## Model types

Each model has a type, which says how it is loaded and which environment its
server runs in. In a YAML it is the model entry's `type`; on the command line,
`cellmap_flow infer <type>`; in the dashboard, the Models tab lists the
CellMap catalog, the `cellmap/*` Hugging Face models and the BioImage Model
Zoo. `cellmap_flow models` lists every type and its arguments.

You rarely need to pick the type yourself: `cellmap_flow add REF` (or the
Models tab's *Add a model*) works it out from what you have, and prints the
YAML entry, saying what it still needs:

```bash
$ cellmap_flow add conscientious-dromedary -v 16      # a BioImage Model Zoo model
$ cellmap_flow add cpsam_v2 -v 64                     # Cellpose-SAM
$ cellmap_flow add cellmap/fly_organelles_run08_438000 # a Hugging Face repo
$ cellmap_flow add /path/to/run/model_checkpoint_20000 --run -d data_path
```

| Type | For | Runs in |
|---|---|---|
| `cellmap` | A folder cellmap_models exported (`metadata.json` + `model.ts`) | this environment |
| `huggingface` | A `cellmap/*` repo on Hugging Face, in that format | this environment |
| `fly` | A fly_organelles training checkpoint, `.ts` or `model.pt` | `fly` (a `.ts`: this environment) |
| `cellpose` | Cellpose 4: Cellpose-SAM (`cpsam_v2`, `cpsam`) or finetuned weights | `cellpose4` |
| `bioimage` | Any BioImage Model Zoo model: id, nickname, DOI, URL or `rdf.yaml` | `bioimageio` |
| `dacapo` | A DaCapo run and iteration | this environment |
| `script` | Any model, from a Python script that defines it | this environment |
| `finetune` | A LoRA adapter or full finetune on top of another model | its base model's |

An entry's `env` overrides where it runs: a pixi environment's name, the path
of a conda environment, or `current` for this one. `~/.cellmap_flow/envs.yaml`
maps environment names to conda environments, for machines without pixi:

```yaml
cellpose4: /groups/mylab/home/me/miniconda3/envs/cellpose4
```

See [YAML configuration](docs/source/yaml_config.rst) for every type's
arguments, and the examples in [example/](example/).

### Cellpose

`type: cellpose` runs Cellpose 4 on each z slice of a chunk. Cellpose-SAM sees
cells best about 30 voxels across, so pick the `voxel_size` (nm) where yours
are about that size:

```yaml
models:
  cells:
    type: cellpose
    voxel_size: 64
    pretrained_model: cpsam_v2   # or cpsam, or the path of finetuned weights
    output: probability          # or masks
```

Masks are made chunk by chunk: add `MortonSegmentationRelabeling` to the
postprocessing so that ids differ between chunks. Cellpose-SAM's weights are
for non-commercial use. See [example/cellpose_sam.yaml](example/cellpose_sam.yaml).

### BioImage Model Zoo

`type: bioimage` runs a [BioImage Model Zoo](https://bioimage.io) model
through bioimageio.core, with the model's own pre- and postprocessing. Its
tile size, halo, output type and (when the model declares units) voxel size
come from the model's description. The zoo's EM models declare no units, so
give `voxel_size`, and a `context` of about 16 voxels to hide seams between
chunks:

```yaml
models:
  mito:
    type: bioimage
    model: conscientious-dromedary   # a 3D mitochondria U-Net
    voxel_size: 16
    context: [0, 16, 16]
```

In the dashboard, the Models tab's *BioImage Model Zoo* list searches the
whole zoo (EM models by default). Models that need prompts or several inputs
(micro-SAM) are not supported. See [example/bioimage_em.yaml](example/bioimage_em.yaml).

### fly_organelles checkpoints

`type: fly` serves a checkpoint fly_organelles trained. Channel names, voxel
sizes and tile size are read from the run's folder (its `train.py`, training
snapshots or `config.yaml`) when the entry does not give them, so the
checkpoint alone is often enough:

```bash
cellmap_flow infer fly -c /path/to/run/model_checkpoint_20000 -d data_path
```

A folder cellmap_models exported is served with `type: cellmap` instead.

### DaCapo

```bash
cellmap_flow infer dacapo -r 20241204_finetune_mito_affs_task_datasplit_v3_u21_kidney_mito_default_cache_8_1 -i 700000 -d /nrs/cellmap/data/jrc_ut21-1413-003/jrc_ut21-1413-003.zarr/recon-1/em/fibsem-uint8/s0
```

### Custom script

This enables using any model by providing a script e.g. [example/model_spec.py](example/model_spec.py)
e.g.
```bash
cellmap_flow infer script -s /groups/cellmap/cellmap/zouinkhim/cellmap-flow/example/model_spec.py -d /nrs/cellmap/data/jrc_mus-cerebellum-1/jrc_mus-cerebellum-1.zarr/recon-1/em/fibsem-uint8/s0
```

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

To run TensorFlow models, we suggest installing TensorFlow via conda: `conda install tensorflow-gpu==2.16.1`

## Run multiple models at once:
List them in a YAML file ([docs/source/yaml_config.rst](docs/source/yaml_config.rst)) and run
```bash
cellmap_flow yaml config.yaml
```
