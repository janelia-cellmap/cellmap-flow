import neuroglancer
import itertools
import logging
import os

neuroglancer.set_server_bind_address("0.0.0.0")

logger = logging.getLogger(__name__)

from cellmap_flow.image_data_interface import ImageDataInterface
from cellmap_flow.utils.scale_pyramid import ScalePyramid


# TODO support multiresolution datasets
def get_raw_layer(dataset_path, filetype, is_multiscale=False):
    if filetype == "n5":
        axis = ["x", "y", "z"]
    else:
        axis = ["z", "y", "x"]
    layers = []

    if is_multiscale:
        scales = [
            f for f in os.listdir(dataset_path) if f[0] == "s" and f[1:].isdigit()
        ]
        scales.sort(key=lambda x: int(x[1:]))
        for scale in scales:
            image = ImageDataInterface(f"{os.path.join(dataset_path, scale)}")
            layers.append(
                neuroglancer.LocalVolume(
                    data=image.ts,
                    dimensions=neuroglancer.CoordinateSpace(
                        names=axis,
                        units="nm",
                        scales=image.voxel_size,
                    ),
                    voxel_offset=image.offset,
                )
            )
        return ScalePyramid(layers)
    else:
        image = ImageDataInterface(dataset_path)
        return neuroglancer.ImageLayer(
            source=neuroglancer.LocalVolume(
                data=image.ts,
                dimensions=neuroglancer.CoordinateSpace(
                    names=axis,
                    units="nm",
                    scales=image.voxel_size,
                ),
                voxel_offset=image.offset,
            )
        )


def generate_neuroglancer_link(dataset_path, inference_dict):
    # Create a new viewer
    viewer = neuroglancer.UnsynchronizedViewer()

    # Add a layer to the viewer
    with viewer.txn() as s:
        is_multi_scale = False
        # if multiscale dataset
        if (
            dataset_path.split("/")[-1].startswith("s")
            and dataset_path.split("/")[-1][1:].isdigit()
        ):
            dataset_path = dataset_path.rsplit("/", 1)[0]
            is_multi_scale = True

        if ".zarr" in dataset_path:
            filetype = "zarr"
        elif ".n5" in dataset_path:
            filetype = "n5"
        else:
            filetype = "precomputed"
        if dataset_path.startswith("/"):
            layer = get_raw_layer(dataset_path, filetype, is_multi_scale)
            s.layers.append("raw", layer)
            # if "nrs/cellmap" in dataset_path:
            #     security = "https"
            #     dataset_path = dataset_path.replace("/nrs/cellmap/", "nrs/")
            # elif "/groups/cellmap/cellmap" in dataset_path:
            #     security = "http"
            #     dataset_path = dataset_path.replace("/groups/cellmap/cellmap/", "dm11/")
            # else:
            #     raise ValueError(
            #         "Currently only supporting nrs/cellmap and /groups/cellmap/cellmap"
            #     )

            # s.layers["raw"] = neuroglancer.ImageLayer(
            #     source=f"{filetype}://{security}://cellmap-vm1.int.janelia.org/{dataset_path}",
            # )
        else:
            s.layers["raw"] = neuroglancer.ImageLayer(
                source=f"{filetype}://{dataset_path}",
            )
        colors = [
            "red",
            "green",
            "blue",
            "yellow",
            "purple",
            "orange",
            "cyan",
            "magenta",
        ]
        color_cycle = itertools.cycle(colors)
        for host, model in inference_dict.items():
            color = next(color_cycle)
            s.layers[model] = neuroglancer.ImageLayer(
                source=f"n5://{host}/{model}",
                shader=f"""#uicontrol invlerp normalized(range=[0, 255], window=[0, 255]);
#uicontrol vec3 color color(default="{color}");
void main(){{emitRGB(color * normalized());}}""",
            )
        # print(viewer)  # neuroglancer.to_url(viewer.state))
        show(str(viewer))
        # logger.error(f"\n \n \n link : {viewer}")
        while True:
            pass


def show(viewer):
    print()
    print()
    print("**********************************************")
    print("LINK:")
    print(viewer)
    print("**********************************************")
    print()
    print()
    print("\033[32mPress Ctrl+C to exit\033[0m")
