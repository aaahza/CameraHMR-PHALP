import torch.serialization
from omegaconf.dictconfig import DictConfig
from omegaconf.base import ContainerMetadata

# Add both DictConfig and ContainerMetadata to the list of safe globals
torch.serialization.add_safe_globals([DictConfig, ContainerMetadata])