import torch
from functools import partial
import torch.serialization
import lightning_lite.utilities.cloud_io

# Method 1: Override torch.load directly to disable weights_only
original_torch_load = torch.load
torch.load = lambda *args, **kwargs: original_torch_load(*args, **{**kwargs, 'weights_only': False})

# Method 2: Override the pytorch_lightning loading function that calls torch.load
original_pl_load = lightning_lite.utilities.cloud_io._load
def patched_pl_load(checkpoint_path, map_location=None):
    return original_torch_load(checkpoint_path, map_location=map_location, weights_only=False)
lightning_lite.utilities.cloud_io._load = patched_pl_load