## Still in development
![CellMapFlow Logo](img/CMFLO_dark.png)

### Real-time inference is performed using Torch/Tensorflow, Dacapo, and bioimage models on local data or any cloud-hosted data.


```bash
$ cellmap_flow

Usage: cellmap_flow [OPTIONS] COMMAND [ARGS]...

  Examples:     
    To use Dacapo run the following commands:  
    cellmap_flow dacapo -r my_run -i iteration -d data_path

    To use custom script
    cellmap_flow script -s script_path -d data_path

    To use bioimage-io model 
    cellmap_flow bioimage -m model_path -d data_path


Commands:
  bioimage  Run the CellMapFlow server with a bioimage-io model.
  dacapo    Run the CellMapFlow server with a DaCapo model.
  script    Run the CellMapFlow server with a custom script.
```


Currently available:
## Using custom script:
which enable using any model by providing a script e.g. [example/model_spec.py](example/model_spec.py)
e.g.
```bash
cellmap_flow script -s /groups/cellmap/cellmap/zouinkhim/cellmap-flow/example/model_spec.py -d /nrs/cellmap/data/jrc_mus-cerebellum-1/jrc_mus-cerebellum-1.zarr/recon-1/em/fibsem-uint8/s0 
```

## Using Dacapo model:
which enable inference using a Dacapo model by providing the run name and iteration number
e.g.
```bash
cellmap_flow dacapo -r 20241204_finetune_mito_affs_task_datasplit_v3_u21_kidney_mito_default_cache_8_1 -i 700000 -d /nrs/cellmap/data/jrc_ut21-1413-003/jrc_ut21-1413-003.zarr/recon-1/em/fibsem-uint8/s0
```

## Using bioimage-io model:
still in development

## Using TensorFlow model:
To run TensorFlow models, we suggest installing TensorFlow via conda: `conda install tensorflow-gpu==2.16.1`

##  Run multiple model at once: 
```bash
cellmap_flow_multiple --script -s /groups/cellmap/cellmap/zouinkhim/cellmap-flow/example/model_spec.py -n script_base --dacapo -r 20241204_finetune_mito_affs_task_datasplit_v3_u21_kidney_mito_default_cache_8_1 -i 700000 -n using_dacapo -d /nrs/cellmap/data/jrc_ut21-1413-003/jrc_ut21-1413-003.zarr/recon-1/em/fibsem-uint8/s0
```


## Limitation:
Currently only supporting data locating in /nrs/cellmap or /groups/cellmap because there is a data server already implemented for them.


