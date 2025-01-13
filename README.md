## Still in development

Currently only `ScriptModelConfig` is working 


First working version
```bash
cellmap_flow -d /nrs/cellmap/data/jrc_mus-cerebellum-1/jrc_mus-cerebellum-1.zarr/recon-1/em/fibsem-uint8/s0 -c /groups/cellmap/cellmap/zouinkhim/cellmap-flow/example/model_spec.py
```

to test the server part - not needed to run an app. happens in the background:
```bash
cellmap_flow_server -d /nrs/cellmap/data/jrc_mus-cerebellum-1/jrc_mus-cerebellum-1.zarr/recon-1/em/fibsem-uint8/s0 -c /groups/cellmap/cellmap/zouinkhim/cellmap-flow/example/model_spec.py
```

TODO :
- bioimage make it to work again
- dacapo add missing cases and run it