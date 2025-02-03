# %%

input_array_info = {"shape": (216, 216, 216), "scale": (8, 8, 8)}
target_array_info = {
    "shape": (68, 68, 68),
    "scale": (4, 4, 4),
    "channels": 14,
    "classes": {
        i: v
        for i, v in enumerate(
            [
                "ecs",
                "pm",
                "mito",
                "mito_mem",
                "ves",
                "ves_mem",
                "endo",
                "endo_mem",
                "er",
                "er_mem",
                "eres",
                "nuc",
                "mt",
                "mt_out",
            ]
        )
    },
}

# %%
import torch
import cellmap_models.pytorch.cosem as cosem_models

model = cosem_models.load_model("setup04/1820500")

# %%
if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

model.to(device)
model.eval()
print("device:", device)
# # %%
