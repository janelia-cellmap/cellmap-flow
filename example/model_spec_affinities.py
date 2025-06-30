#%%
# pip install fly-organelles
from funlib.geometry.coordinate import Coordinate
import torch
import funlib.learn.torch
import numpy as np
voxel_size = (16, 16, 16)
read_shape = Coordinate((178, 178, 178)) * Coordinate(voxel_size)
write_shape = Coordinate((56, 56, 56)) * Coordinate(voxel_size)
output_voxel_size = Coordinate((16, 16, 16))

#%%
class StandardUnet(torch.nn.Module):
    def __init__(
        self,
        out_channels,
        num_fmaps=16,
        fmap_inc_factor=6,
        downsample_factors=None,
        kernel_size_down=None,
        kernel_size_up=None,
    ):
        super().__init__()
        if downsample_factors is None:
            downsample_factors = [(2, 2, 2), (2, 2, 2), (2, 2, 2)]
        if kernel_size_down is None:
            kernel_size_level = [(3, 3, 3), (3, 3, 3), (3, 3, 3)]
            kernel_size_down = [
                kernel_size_level,
            ] * (len(downsample_factors) + 1)
        if kernel_size_up is None:
            kernel_size_level = [(3, 3, 3), (3, 3, 3), (3, 3, 3)]
            kernel_size_level = [
                kernel_size_level,
            ] * len(downsample_factors)

        self.unet_backbone = funlib.learn.torch.models.UNet(
            in_channels=1,
            num_fmaps=num_fmaps,
            fmap_inc_factor=fmap_inc_factor,
            downsample_factors=downsample_factors,
            kernel_size_down=kernel_size_down,
            constant_upsample=True,
        )

        self.final_conv = torch.nn.Conv3d(num_fmaps, out_channels, (1, 1, 1), padding="valid")

    def forward(self, raw):
        x = self.unet_backbone(raw)
        return self.final_conv(x)
#%%
def load_eval_model(num_labels, checkpoint_path):
    model_backbone = StandardUnet(num_labels)
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print("device:", device)    
    checkpoint = torch.load(checkpoint_path, weights_only=True, map_location=device)
    model_backbone.load_state_dict(checkpoint["model_state_dict"])
    model = torch.nn.Sequential(model_backbone, torch.nn.Sigmoid())
    model.to(device)
    model.eval()
    return model


classes = ["mito",]*9
CHECKPOINT_PATH = "/groups/cellmap/cellmap/zouinkhim/c-elegen/v2/train/fly_run/all/affinities/new/run04_mito/model_checkpoint_65000"
output_channels = len(classes)
model = load_eval_model(output_channels, CHECKPOINT_PATH)
block_shape = np.array((56, 56, 56,output_channels))
# %%
