from pathlib import Path
from pprint import pprint
from typing import Any

import imageio
import matplotlib.pyplot as plt
import numpy as np
import torch
from numpy.typing import NDArray

from bioimageio.spec.model.v0_5 import (
    Author,
    AxisId,
    BatchAxis,
    ChannelAxis,
    CiteEntry,
    DatasetId,
    Doi,
    FileDescr,
    HttpUrl,
    Identifier,
    InputTensorDescr,
    IntervalOrRatioDataDescr,
    LicenseId,
    LinkedDataset,
    ModelDescr,
    OrcidId,
    ParameterizedSize,
    PytorchStateDictWeightsDescr,
    ScaleRangeDescr,
    ScaleRangeKwargs,
    SpaceInputAxis,
    TensorId,
    TorchscriptWeightsDescr,
    WeightsDescr,
)
from bioimageio.spec.pretty_validation_errors import (
    enable_pretty_validation_errors_in_ipynb,
)


# %%
enable_pretty_validation_errors_in_ipynb()
CMAP = "Greys"


def show(*paths: Path, d: int = 2):
    fig, axes = plt.subplots(1, len(paths), squeeze=False)
    for path, ax in zip(paths, axes.flatten()):
        ax.set_title(path.name)
        if path.suffix == ".npy":
            img: NDArray[Any] = np.load(path)
        else:
            img = imageio.v2.imread(path)

        img = img.squeeze()

        if img.ndim > 3 or img.ndim == 3 and min(img.shape) > 4:
            show3d(ax, img, d=d)
        else:
            show2d(ax, img)

    plt.show()


def show2d(ax, img: NDArray[Any]):
    ax.imshow(img, cmap=CMAP, vmin=0, vmax=1)
    ax.set_axis_off()


def show3d(ax, img: NDArray[Any], d: int):
    z, y, x = img.shape
    img -= np.percentile(img, 20)
    img = img / np.percentile(img, 99.99)

    img_show = np.ones((y + d + z, x + d + z), dtype=img.dtype)
    for a in (0, 2, 1):
        img_part = img.squeeze().max(axis=a)
        if a == 0:
            assert img_part.shape == (y, x), img_part.shape
            img_show[:y, :x] = img_part
        elif a == 1:
            assert img_part.shape == (z, x), img_part.shape
            img_show[y + d :, :x] = img_part
        elif a == 2:
            img_part = img_part.T
            assert img_part.shape == (y, z), img_part.shape
            img_show[:y, x + d :] = img_part
        else:
            raise NotImplementedError

    img_show[y + d :, x + d :] = 0
    # print(img_show.min(), img_show.max())
    ax.imshow(img_show, cmap=CMAP, vmin=0, vmax=1)
    ax.set_axis_off()


# %%
# Description details
pytorch_weights = torch.load(root / "weights.pt", weights_only=False)
pprint([(k, tuple(v.shape)) for k, v in pytorch_weights.items()][:4] + ["..."])
