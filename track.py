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

def expand_to_aspect_ratio(input_shape, target_aspect_ratio=None):
    """Increase the size of the bounding box to match the target shape."""
    if target_aspect_ratio is None:
        return input_shape

    try:
        w , h = input_shape
    except (ValueError, TypeError):
        return input_shape

    w_t, h_t = target_aspect_ratio
    if h / w < h_t / w_t:
        h_new = max(w * h_t / w_t, h)
        w_new = w
    else:
        h_new = h
        w_new = max(h * w_t / h_t, w)
    if h_new < h or w_new < w:
        breakpoint()
    return np.array([w_new, h_new])

def expand_bbox_to_aspect_ratio(bbox, target_aspect_ratio=None):
    # bbox: np.array: (N,4) detectron2 bbox format 
    # target_aspect_ratio: (width, height)
    if target_aspect_ratio is None:
        return bbox
    
    is_singleton = (bbox.ndim == 1)
    if is_singleton:
        bbox = bbox[None,:]

    if bbox.shape[0] > 0:
        center = np.stack(((bbox[:,0] + bbox[:,2]) / 2, (bbox[:,1] + bbox[:,3]) / 2), axis=1)
        scale_wh = np.stack((bbox[:,2] - bbox[:,0], bbox[:,3] - bbox[:,1]), axis=1)
        scale_wh = np.stack([expand_to_aspect_ratio(wh, target_aspect_ratio) for wh in scale_wh], axis=0)
        bbox = np.stack([
            center[:,0] - scale_wh[:,0] / 2,
            center[:,1] - scale_wh[:,1] / 2,
            center[:,0] + scale_wh[:,0] / 2,
            center[:,1] + scale_wh[:,1] / 2,
        ], axis=1)

    if is_singleton:
        bbox = bbox[0,:]

    return bbox

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
    
class HMR2023TextureSampler(CameraHMRPredictor):
    def __init__(self, cfg) -> None:
        super().__init__(cfg)

        # Model's all set up. Now, load tex_bmap and tex_fmap
        # Texture map atlas
        bmap = np.load(os.path.join(CACHE_DIR, 'phalp/3D/bmap_256.npy'))
        fmap = np.load(os.path.join(CACHE_DIR, 'phalp/3D/fmap_256.npy'))
        self.register_buffer('tex_bmap', torch.tensor(bmap, dtype=torch.float))
        self.register_buffer('tex_fmap', torch.tensor(fmap, dtype=torch.long))

        self.img_size = 256         #self.cfg.MODEL.IMAGE_SIZE
        # self.focal_length = 5000.   #self.cfg.EXTRA.FOCAL_LENGTH

        import neural_renderer as nr
        self.neural_renderer = nr.Renderer(dist_coeffs=None, orig_size=self.img_size,
                                          image_size=self.img_size,
                                          light_intensity_ambient=1,
                                          light_intensity_directional=0,
                                          anti_aliasing=False)

    def forward(self, x):
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

        pred_smpl_params, pred_cam, fl_h = self.model(batch)

        def unproject_uvmap_to_mesh(bmap, fmap, verts, faces):
            # bmap:  256,256,3
            # fmap:  256,256
            # verts: B,V,3
            # faces: F,3
            valid_mask = (fmap >= 0)

            fmap_flat = fmap[valid_mask]      # N
            bmap_flat = bmap[valid_mask,:]    # N,3

            face_vids = faces[fmap_flat, :]  # N,3
            face_verts = verts[:, face_vids, :] # B,N,3,3

            bs = face_verts.shape
            map_verts = torch.einsum('bnij,ni->bnj', face_verts, bmap_flat) # B,N,3

            return map_verts, valid_mask
        
        
        pred_vertices, _, cam_trans = self.estimator.get_output_mesh(pred_smpl_params, pred_cam, batch)
        pred_verts = pred_vertices + cam_trans.unsqueeze(1)
        device = pred_verts.device
        face_tensor = torch.tensor(self.smpl.faces.astype(np.int64), dtype=torch.long, device=device)
        map_verts, valid_mask = unproject_uvmap_to_mesh(self.tex_bmap, self.tex_fmap, pred_verts, face_tensor) # B,N,3

        # Project map_verts to image using K,R,t
        # map_verts_view = einsum('bij,bnj->bni', R, map_verts) + t # R=I t=0
        focal = fl_h / (self.img_size / 2)
        map_verts_proj = focal * map_verts[:, :, :2] / map_verts[:, :, 2:3] # B,N,2
        map_verts_depth = map_verts[:, :, 2] # B,N
      
        # Render Depth. Annoying but we need to create this
        K = torch.eye(3, device=device)
        K[0, 0] = K[1, 1] = fl_h
        K[1, 2] = K[0, 2] = self.img_size / 2  # Because the neural renderer only support squared images
        K = K.unsqueeze(0)
        R = torch.eye(3, device=device).unsqueeze(0)
        t = torch.zeros(3, device=device).unsqueeze(0)
        rend_depth = self.neural_renderer(pred_verts,
                                        face_tensor[None].expand(pred_verts.shape[0], -1, -1).int(),
                                        # textures=texture_atlas_rgb,
                                        mode='depth',
                                        K=K, R=R, t=t)

        rend_depth_at_proj = torch.nn.functional.grid_sample(rend_depth[:,None,:,:], map_verts_proj[:,None,:,:]) # B,1,1,N
        rend_depth_at_proj = rend_depth_at_proj.squeeze(1).squeeze(1) # B,N

        img_rgba = torch.cat([batch['img'], batch['mask'][:,None,:,:]], dim=1) # B,4,H,W
        img_rgba_at_proj = torch.nn.functional.grid_sample(img_rgba, map_verts_proj[:,None,:,:]) # B,4,1,N
        img_rgba_at_proj = img_rgba_at_proj.squeeze(2) # B,4,N

        visibility_mask = map_verts_depth <= (rend_depth_at_proj + 1e-4) # B,N
        img_rgba_at_proj[:,3,:][~visibility_mask] = 0

        # Paste image back onto square uv_image
        uv_image = torch.zeros((batch['img'].shape[0], 4, 256, 256), dtype=torch.float, device=device)
        uv_image[:, :, valid_mask] = img_rgba_at_proj

        out = {
            'uv_image':  uv_image,
            'uv_vector' : self.hmar_old.process_uv_image(uv_image),
            'pose_smpl': pred_smpl_params,
            'pred_cam':  cam_trans,
        }
        return out

class HMR2_4dhuman(PHALP):
    def __init__(self, cfg):
        super().__init__(cfg)

    def setup_hmr(self): 
        self.HMAR = HMR2023TextureSampler(self.cfg)

    def get_detections(self, image, frame_name, t_, additional_data=None, measurments=None):
        (
            pred_bbox, pred_bbox, pred_masks, pred_scores, pred_classes, 
            ground_truth_track_id, ground_truth_annotations
        ) =  super().get_detections(image, frame_name, t_, additional_data, measurments)

        # Pad bounding boxes 
        pred_bbox_padded = expand_bbox_to_aspect_ratio(pred_bbox, self.cfg.expand_bbox_shape)

        return (
            pred_bbox, pred_bbox_padded, pred_masks, pred_scores, pred_classes,
            ground_truth_track_id, ground_truth_annotations
        )
    
class PHALP_Prime_TokenHMR(PHALP):
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

    phalp_tracker = PHALP_Prime_TokenHMR(cfg)

    phalp_tracker.track()

if __name__ == "__main__":
    main()
