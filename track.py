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
from core.datasets.dataset import Dataset
from core.utils import recursive_to

warnings.filterwarnings("ignore")

log = get_pylogger(__name__)

class CameraHMRPredictor(HMR2018Predictor):
    def __init__(self, cfg) -> None:
        super().__init__(cfg)

        self.estimator = HumanMeshEstimator()

        self.model = self.estimator.model
        self.model.eval()

    def forward(self, x):
        # Get the base output from the parent class
        hmar_out = self.hmar_old(x)
        device = x.device
        batch_size = x.size(0)
        
        # Process each image in the batch
        all_smpl_params = []
        all_pred_cam = []
        all_focal_lengths = []
        
        for b in range(batch_size):
            # Extract the image and mask for this batch item
            img_tensor = x[b, :3]  # [3, H, W]
            mask = x[b, 3:4] if x.size(1) > 3 else None  # Get mask if available
            
            # Convert to numpy for detector
            img_cv2 = img_tensor.detach().cpu().numpy().transpose(1, 2, 0)  # [H, W, 3]
            img_h, img_w = img_cv2.shape[:2]
            
            # Run detector
            det_out = self.estimator.detector(img_cv2)
            det_instances = det_out['instances']
            valid_idx = (det_instances.pred_classes == 0) & (det_instances.scores > 0.5)
            
            if valid_idx.sum() == 0:
                # If no detections, create default parameters
                default_smpl_params = {
                    'global_orient': torch.zeros((1, 1, 3, 3), device=device),
                    'body_pose': torch.zeros((1, 21, 3, 3), device=device),
                    'betas': torch.zeros((1, 10), device=device),
                }
                default_cam = torch.zeros((1, 3), device=device)
                default_focal_length = torch.tensor([img_h], device=device).float()
                
                all_smpl_params.append(default_smpl_params)
                all_pred_cam.append(default_cam)
                all_focal_lengths.append(default_focal_length)
                continue
                
            # Process valid detections
            boxes = det_instances.pred_boxes.tensor[valid_idx].cpu().numpy()
            bbox_scale = (boxes[:, 2:4] - boxes[:, 0:2]) / 200.0 
            bbox_center = (boxes[:, 2:4] + boxes[:, 0:2]) / 2.0
            
            # Get camera intrinsics
            cam_int = self.estimator.get_cam_intrinsics(img_cv2)
            
            # Create dataset and dataloader for this image
            dataset = Dataset(img_cv2, bbox_center, bbox_scale, cam_int, False)
            
            # If dataset is empty, use default parameters
            if len(dataset) == 0:
                default_smpl_params = {
                    'global_orient': torch.zeros((1, 1, 3, 3), device=device),
                    'body_pose': torch.zeros((1, 21, 3, 3), device=device),
                    'betas': torch.zeros((1, 10), device=device),
                }
                default_cam = torch.zeros((1, 3), device=device)
                default_focal_length = torch.tensor([img_h], device=device).float()
                
                all_smpl_params.append(default_smpl_params)
                all_pred_cam.append(default_cam)
                all_focal_lengths.append(default_focal_length)
                continue
            
            # Create dataloader with appropriate batch size
            dataloader = torch.utils.data.DataLoader(
                dataset, 
                batch_size=len(dataset), 
                shuffle=False, 
                num_workers=0  # Use 0 for debugging, increase for performance
            )
            
            # Process the batch
            batch = next(iter(dataloader))
            batch = recursive_to(batch, device)
            
            # Run the model
            with torch.no_grad():
                out_smpl_params, out_cam, focal_length_ = self.model(batch)
            
            # Store results
            all_smpl_params.append(out_smpl_params)
            all_pred_cam.append(out_cam)
            all_focal_lengths.append(focal_length_)
        
        # Use the first detection for each image (or default if none)
        # This matches the way PHALP typically works with one person per frame
        smpl_params = {
            'global_orient': torch.cat([p['global_orient'][0:1] for p in all_smpl_params], dim=0),
            'body_pose': torch.cat([p['body_pose'][0:1] for p in all_smpl_params], dim=0),
            'betas': torch.cat([p['betas'][0:1] for p in all_smpl_params], dim=0),
        }
        pred_cam = torch.cat([cam[0:1] for cam in all_pred_cam], dim=0)
        focal_length = torch.cat([fl[0:1] for fl in all_focal_lengths], dim=0)
        
        # Combine with original output
        out = hmar_out | {
            'pose_smpl': smpl_params,
            'pred_cam': pred_cam,
            'focal_length': focal_length
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
