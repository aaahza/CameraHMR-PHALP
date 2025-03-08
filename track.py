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
        hmar_out = self.hmar_old(x)
        device = x.device

        img_tensor = x[0, :3]  # [3, H, W]
        img_cv2 = img_tensor.detach().cpu().numpy().transpose(1, 2, 0)  # [H, W, 3]

        det_out = self.estimator.detector(img_cv2)
        det_instances = det_out['instances']
        valid_idx = (det_instances.pred_classes == 0) & (det_instances.scores > 0.5)
        boxes = det_instances.pred_boxes.tensor[valid_idx].cpu().numpy()
        bbox_scale = (boxes[:, 2:4] - boxes[:, 0:2]) / 200.0 
        bbox_center = (boxes[:, 2:4] + boxes[:, 0:2]) / 2.0

        cam_int = self.estimator.get_cam_intrinsics(img_cv2)

        dataset = Dataset(img_cv2, bbox_center, bbox_scale, cam_int, False)
        batch_size = len(dataset)
        if batch_size == 0:
            raise ValueError("No valid detections found, dataset is empty.")

        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=10)
        
        batch = next(iter(dataloader))
        batch = recursive_to(batch, device)
        img_h, img_w = batch['img_size'][0]
        with torch.no_grad():
            out_smpl_params, out_cam, focal_length_ = self.model(batch)

        # out = {
        #     **hmar_out,
        #     'pose_smpl': out_smpl_params,
        #     'pred_cam': out_cam,
        #     'focal_length': focal_length_,
        #     # 'output_vertices': output_vertices,
        #     # 'output_joints': output_joints,
        #     # 'output_cam_trans': output_cam_trans
        # }

        # pred_smpl_params, pred_cam, _ = self.model(batch)

        out = hmar_out | {
            'pose_smpl': out_smpl_params,
            'pred_cam': out_cam,
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
