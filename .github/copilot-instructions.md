# Review guidance for cellmap-flow

cellmap-flow serves PyTorch, TensorFlow, DaCapo and bioimage.io models to Neuroglancer as virtual zarr arrays, runs them blockwise over whole datasets, and finetunes them from annotations painted in the dashboard. Check changes against these conventions.

**Geometry**
- World coordinates are in nm. A `Roi` is (lower corner, shape) in nm, and voxel `i` covers `[corner + i*vs, corner + (i+1)*vs)` along each axis.
- An OME-NGFF `translation` is the centre of voxel 0, not its corner: read it with `io.ome.ome_corner` and write it with `io.ome.ome_translation`. Legacy `resolution`/`offset` attributes and N5 `transform` are already corners.
- Voxel sizes can be fractional (5.24 nm), and a corner need not lie on the voxel grid (Janelia raw has a -4 nm corner at 8 nm). `Coordinate` truncates to integers, so use `io.geometry` (`Grid`, `coordinate_or_floats`) or keep floats.
- A model's forward takes `(batch, 1, z, y, x)` and returns `(batch, C, z, y, x)`. A processed chunk is `(c, z, y, x)` unless the config sets `chunk_output_axes`; the served zarr array puts the channel axis last.

**Code**
- `peft` and `tensorboard` come from the optional `finetune` extra, so import them inside the functions that use them, never at module level. CI runs the tests without that extra too.
- Never `eval` or `exec` a string from a request, a layer URL or a YAML file; Lambda expressions go through `utils/safe_expression.compile_expression`. Inference servers and the dashboard listen on 0.0.0.0, so anything they receive can come from any host on the network.
- Give `subprocess` an argv list, and `shlex.quote` every value put into a command string such as a bsub command.
- Library code raises (`ConfigError` for a bad configuration); only CLI entry points call `sys.exit`.
- New code does not add state to the `Flow` singleton `g` (`globals.py`) or hard-code site values such as `gpu_h100`, 8 nm or `sN` level names; take them as arguments or from the data's metadata.
- Comments and docstrings explain why, in plain prose: the constraint, the bug a line prevents, the units. They do not narrate the code.

**Tests**
- Tests assert on behaviour (return values, files written, HTTP responses), not on source text (`inspect.getsource`), signatures, private attributes or exact log wording.
- A test that needs peft is marked `@pytest.mark.finetune` and skips without it; `gpu`, `lsf`, `minio` and `network` work the same way. Write files under `tmp_path`.
- A bug fix comes with a test that fails without it.
