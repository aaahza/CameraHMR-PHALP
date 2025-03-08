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
        
        # Import needed transforms directly
        from torchvision.transforms import Normalize
        from core.constants import IMAGE_SIZE, IMAGE_MEAN, IMAGE_STD
        self.normalize_img = Normalize(mean=IMAGE_MEAN, std=IMAGE_STD)
        self.img_size = IMAGE_SIZE
        self.mean = np.array(IMAGE_MEAN) * 255.0
        self.std = np.array(IMAGE_STD) * 255.0

    def preprocess_image(self, img_tensor, device):
        """Preprocess the image tensor for the model"""
        from core.utils.utils import (convert_cvimg_to_tensor,
                    expand_to_aspect_ratio,
                    generate_image_patch_cv2)
        import cv2
        
        # Convert tensor to numpy
        img_cv2 = img_tensor.detach().cpu().numpy().transpose(1, 2, 0)  # [H, W, 3]
        img_h, img_w = img_cv2.shape[:2]
        
        # Get camera intrinsics
        cam_int = self.estimator.get_cam_intrinsics(img_cv2)
        
        # Run detector
        det_out = self.estimator.detector(img_cv2)
        det_instances = det_out['instances']
        valid_idx = (det_instances.pred_classes == 0) & (det_instances.scores > 0.5)
        
        if valid_idx.sum() == 0:
            # Return default values if no detections
            return None, None, None, img_h, img_w, cam_int
        
        # Get best detection (highest score)
        best_idx = det_instances.scores[valid_idx].argmax().item() if valid_idx.sum() > 1 else 0
        best_box = det_instances.pred_boxes.tensor[valid_idx][best_idx].cpu().numpy()
        
        # Calculate center and scale
        center = (best_box[2:4] + best_box[0:2]) / 2.0
        scale = (best_box[2:4] - best_box[0:2]) / 200.0
        bbox_size = expand_to_aspect_ratio(scale*200, target_aspect_ratio=None).max()
        
        # Generate image patch
        center_x, center_y = center
        patch_width = patch_height = self.img_size
        img_patch_cv, trans = generate_image_patch_cv2(img_cv2,
                                                    center_x, center_y,
                                                    bbox_size, bbox_size,
                                                    patch_width, patch_height,
                                                    False, 1.0, 0,
                                                    border_mode=cv2.BORDER_CONSTANT)
        
        # Convert to tensor and normalize
        img_patch = convert_cvimg_to_tensor(img_patch_cv[:, :, ::-1])
        
        # Apply normalization
        for n_c in range(min(img_cv2.shape[2], 3)):
            img_patch[n_c, :, :] = (img_patch[n_c, :, :] - self.mean[n_c]) / self.std[n_c]
        
        img_patch = torch.tensor(img_patch, device=device).float()
        
        # Create batch dictionary with required fields
        batch = {
            'img': img_patch.unsqueeze(0),
            'personid': torch.tensor([0], device=device),
            'box_center': torch.tensor([center], device=device),
            'box_size': torch.tensor([bbox_size], device=device),
            'img_size': torch.tensor([[img_h, img_w]], device=device),
            'cam_int': torch.tensor([cam_int], device=device)
        }
        
        return batch, center, bbox_size, img_h, img_w, cam_int

    def forward(self, x):
        # Get the base output from the parent class
        hmar_out = self.hmar_old(x)
        device = x.device
        batch_size = x.size(0)
        
        # Process each image in batch
        all_smpl_params = []
        all_pred_cam = []
        all_focal_lengths = []
        
        for b in range(batch_size):
            # Extract image for this batch item
            img_tensor = x[b, :3]  # [3, H, W]
            
            # Preprocess image directly without using Dataset
            processed_batch, center, bbox_size, img_h, img_w, cam_int = self.preprocess_image(img_tensor, device)
            
            if processed_batch is None:
                # No detections found, use default values
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
            
            # Run inference with model
            with torch.no_grad():
                out_smpl_params, out_cam, focal_length_ = self.model(processed_batch)
            
            # Store results
            all_smpl_params.append(out_smpl_params)
            all_pred_cam.append(out_cam)
            all_focal_lengths.append(focal_length_)
        
        # Handle case where batch is empty (no detections in any image)
        if not all_smpl_params:
            default_smpl_params = {
                'global_orient': torch.zeros((batch_size, 1, 3, 3), device=device),
                'body_pose': torch.zeros((batch_size, 21, 3, 3), device=device),
                'betas': torch.zeros((batch_size, 10), device=device),
            }
            default_cam = torch.zeros((batch_size, 3), device=device)
            default_focal_length = torch.ones((batch_size), device=device).float() * 1000  # Default focal length
            
            # Combine with original output
            out = hmar_out | {
                'pose_smpl': default_smpl_params,
                'pred_cam': default_cam,
                'focal_length': default_focal_length
            }
            return out
        
        # Combine results for the batch
        smpl_params = {
            'global_orient': torch.cat([p['global_orient'] for p in all_smpl_params], dim=0),
            'body_pose': torch.cat([p['body_pose'] for p in all_smpl_params], dim=0),
            'betas': torch.cat([p['betas'] for p in all_smpl_params], dim=0),
        }
        pred_cam = torch.cat([cam for cam in all_pred_cam], dim=0)
        focal_length = torch.cat([fl for fl in all_focal_lengths], dim=0)
        
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
