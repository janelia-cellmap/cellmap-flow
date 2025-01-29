# %%
from cellmap_flow.server import CellMapFlowServer
from cellmap_flow.utils.data import (
    ModelConfig,
    BioModelConfig,
    DaCapoModelConfig,
    ScriptModelConfig,
)
import argparse


def server_check(script_path, dataset):
    model_config = ScriptModelConfig(script_path=script_path)
    server = CellMapFlowServer(dataset, model_config)
    chunk_x = 2
    chunk_y = 2
    chunk_z = 2

    server._chunk_impl(None, None, chunk_x, chunk_y, chunk_z, None)

    print("Server check passed")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test run a CellMapFlow server")
    parser.add_argument(
        "--script_path",
        "-s",
        type=str,
        help="Path to the Python script containing model specification",
    )
    parser.add_argument("--dataset", "-d", type=str, help="Path to the dataset")
    args = parser.parse_args()
    server_check(args.script_path, args.dataset)
