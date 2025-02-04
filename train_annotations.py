import torch
import detectron2
from detectron2.data.datasets import register_coco_instances
from detectron2.engine import DefaultTrainer
from detectron2.config import get_cfg
from detectron2.data import DatasetMapper, build_detection_train_loader
from detectron2.data.transforms import ResizeShortestEdge
import detectron2.data.transforms as T
import os
import cv2
import numpy as np

# Define dataset paths
dataset_name = "corn_kernels"
image_dir = "datasets/images"
depth_dir = "datasets/depth_maps"
json_file = "datasets/annotations/labels_section1annotations.json"

# Register dataset in Detectron2
register_coco_instances(dataset_name, {}, json_file, image_dir)

# Configurations
cfg = get_cfg()
cfg.merge_from_file("configs/mask3d_config.yaml")  # Load pre-trained model
cfg.DATASETS.TRAIN = (dataset_name,)
cfg.DATASETS.TEST = ()  # No validation for now
cfg.DATALOADER.NUM_WORKERS = 0
#cfg.MODEL.WEIGHTS = "detectron2://COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x/137849600/model_final_f10217.pkl"  # Pretrained on COCO
cfg.SOLVER.IMS_PER_BATCH = 2
cfg.SOLVER.BASE_LR = 0.0025  # Adjust learning rate
cfg.SOLVER.MAX_ITER = 5000  # Training iterations
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1  # Only "corn kernel"

# Modify Data Loader to Include Depth Maps
def add_depth_to_image(dataset_dict):
    """Custom function to load RGB + depth images."""
    record = dataset_dict.copy()

    # Ensure correct file path
    img_path = os.path.join(image_dir, os.path.basename(record["file_name"]))
    depth_path = os.path.join(depth_dir, os.path.basename(record["file_name"]))

    # Load RGB image
    img = cv2.imread(img_path)
    if img is None:
        raise FileNotFoundError(f"❌ Error: Image not found at {img_path}")

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_h, img_w, _ = img.shape  # Get image dimensions

    # Load depth map
    depth_map = cv2.imread(depth_path, cv2.IMREAD_GRAYSCALE)
    if depth_map is None:
        raise FileNotFoundError(f"❌ Error: Depth map not found at {depth_path}")

    # Resize depth map to match RGB image dimensions
    depth_map_resized = cv2.resize(depth_map, (img_w, img_h), interpolation=cv2.INTER_LINEAR)
    depth_map_resized = depth_map_resized.astype(np.float32) / 255.0  # Normalize depth
    depth_map_resized = np.expand_dims(depth_map_resized, axis=-1)  # Convert to single-channel

    # Merge depth as a 4th channel
    img_with_depth = np.concatenate((img, depth_map_resized), axis=-1).astype(np.float32)

    record["image"] = torch.tensor(img_with_depth.transpose(2, 0, 1))  # Convert to (C, H, W)
    return record

# Modify Trainer to Use Depth-Enhanced Images
class DepthTrainer(DefaultTrainer):
    def __init__(self, cfg):
        super().__init__(cfg)

    def build_train_loader(self, cfg):
        return build_detection_train_loader(
            cfg,
            mapper=lambda dataset_dict: add_depth_to_image(DatasetMapper(cfg, is_train=True)(dataset_dict))
        )

    # Start Training


os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

if __name__ == "__main__":
    trainer = DepthTrainer(cfg)
    trainer.resume_or_load(resume=False)
    trainer.train()