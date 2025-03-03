import torch.serialization
from omegaconf.dictconfig import DictConfig

# Add DictConfig to the list of safe globals
torch.serialization.add_safe_globals([DictConfig])