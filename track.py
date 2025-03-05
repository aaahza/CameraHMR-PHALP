import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import os
import hydra 
import torch
import numpy as np
from hydra.core.config_store import ConfigStore
from omegaconf import DictConfig

from phalp.configs.base import FullConfig
from phalp.models.hmar.hmr import HMR2018Predictor
from phalp.trackers.PHALP import PHALP
from phalp.utils import get_pylogger
from phalp.configs.base import CACHE_DIR

from mesh_estimator import HumanMeshEstimator

warnings.filterwarnings("ignore")

log = get_pylogger(__name__)

class CameraHMRPredictor(HMR2018Predictor):
    def __init__(self, cfg) -> None:
        super().__init__(cfg)

        self.estimator = HumanMeshEstimator()

        self.model = self.estimator.model
        self.model.eval()

    def forward(self, x):
        hmar_out = self.hmar_old(x)

        B, _, H, W = x.shape
        
        # Create a complete batch with all required fields
        batch = {
            'img': x[:, :3, :, :],                           # RGB channels
            'mask': torch.clamp(x[:, 3, :, :], 0, 1),        # Mask channel
            'box_center': torch.tensor([[W/2, H/2]] * B, device=x.device),  # Center of the image
            'box_size': torch.tensor([max(H, W)] * B, device=x.device),     # Size of the box
            'img_size': torch.tensor([[H, W]] * B, device=x.device),        # Original image size
            'cam_int': torch.tensor([[                        # Default camera intrinsics
                [5000.0, 0.0, W/2],
                [0.0, 5000.0, H/2],
                [0.0, 0.0, 1.0]
            ]] * B, device=x.device)
        }

        pred_smpl_params, pred_cam, _ = self.model(batch)
        out = hmar_out | {
            'pose_smpl': pred_smpl_params,
            'pred_cam': pred_cam,
        }
        return out
    
    
class PHALP_CameraHMR(PHALP):
    def __init__(self, cfg):
        super().__init__(cfg)

    def setup_hmr(self):
        self.HMAR = CameraHMRPredictor(self.cfg)


@dataclass
class Human4DConfig(FullConfig):
    # override defaults if needed
    expand_bbox_shape: Optional[Tuple[int]] = (192,256)
    pass

cs = ConfigStore.instance()
cs.store(name="config", node=Human4DConfig)

@hydra.main(version_base="1.2", config_name="config")
def main(cfg: DictConfig) -> Optional[float]:
    """Main function for running the PHALP tracker."""

    phalp_tracker = PHALP_CameraHMR(cfg)

    phalp_tracker.track()

if __name__ == "__main__":
    main()
