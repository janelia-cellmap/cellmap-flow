"""The finetune tab's geometry keeps fractional voxel sizes from model_info."""

from cellmap_flow.utils.server_info import model_geometry


def test_fractional_voxel_sizes_are_kept_and_whole_ones_stay_ints():
    geometry = model_geometry(
        {"write_shape": [448.0, 448, 448], "output_voxel_size": [5.24, 8, 8.0], "output_channels": 2}
    )
    assert geometry == {
        "write_shape": [448, 448, 448],
        "output_voxel_size": [5.24, 8, 8],
        "output_channels": 2,
    }
    assert all(isinstance(v, int) for v in geometry["write_shape"])


def test_a_payload_without_geometry_gives_none():
    assert model_geometry({"output_activation": "sigmoid"}) is None
