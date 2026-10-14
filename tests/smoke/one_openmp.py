"""Exits 0 when numpy and torch share one OpenMP runtime in this environment.

Run with an environment's python, not by pytest (test_smoke_scripts runs it
in a subprocess, and the macOS CI job in each pixi environment). conda-forge's
Mac OpenBLAS was its OpenMP build, which loads llvm-openmp's libomp with
numpy, while PyTorch's PyPI wheel loads its own: with both in one process,
finetuning on a Mac aborted with "OMP: Error #15: Initializing libomp.dylib,
but found libomp.dylib already initialized". Both libraries' threaded paths
run here, so a second runtime aborts this process the same way.
"""

import numpy as np
import torch

a = np.random.default_rng(0).random((1024, 1024))
assert np.isfinite(a @ a).all()
t = torch.rand(1024, 1024)
assert torch.isfinite(t @ t).all()
print(f"numpy {np.__version__} and torch {torch.__version__} in one process: ok")
