import torch
import detectron2
from detectron2.data.datasets import register_coco_instances
from detectron2.engine import DefaultTrainer
from detectron2.config import get_cfg
from detectron2.evaluation import COCOEvaluator
from detectron2.evaluation import inference_on_dataset
from detectron2.data import build_detection_test_loader
from detectron2.data import transforms as T
from detectron2.data import DatasetMapper
import os

# Define dataset paths
dataset_name = "corn_kernels"
image_dir = "datasets/images"
json_file = "datasets/annotations/labels_section1annotations.json"

# Register dataset in Detectron2
register_coco_instances(dataset_name, {}, json_file, image_dir)

# Configurations
cfg = get_cfg()
cfg.merge_from_file("detectron2/configs/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")  # Load pre-trained model
cfg.DATASETS.TRAIN = (dataset_name,)
cfg.DATASETS.TEST = ()  # No validation for now
cfg.DATALOADER.NUM_WORKERS = 0
cfg.MODEL.WEIGHTS = "detectron2://COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x/137849600/model_final_f10217.pkl"  # Pretrained on COCO
cfg.SOLVER.IMS_PER_BATCH = 2
cfg.SOLVER.BASE_LR = 0.0025  # Adjust learning rate
cfg.SOLVER.MAX_ITER = 1000  # Training iterations
cfg.MODEL.RPN.POST_NMS_TOPK_TRAIN = 1000  # Increase region proposals
cfg.MODEL.RPN.POST_NMS_TOPK_TEST = 1000
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128  # Increase training batch size
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1  # Only corn kernels
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7  # CONFIDENCE : Increase threshold for better accuracy
cfg.MODEL.DEVICE = "cuda"  # Ensure training uses GPU
#cfg.MODEL.ANCHOR_GENERATOR.SIZES = [[16, 32, 64, 128, 256]]  # Adjust anchor sizes
#cfg.MODEL.ANCHOR_GENERATOR.ASPECT_RATIOS = [[0.5, 1.0, 2.0]]  # Aspect ratios

# Define augmentation pipeline
# augmentation = [
#     T.RandomBrightness(0.8, 1.2),  # Adjust brightness
#     T.RandomFlip(prob=0.5, horizontal=True, vertical=False),  # Horizontal flip
#     T.RandomRotation(angle=[-10, 10])  # Random rotation
# ]

cfg.DATALOADER.FILTER_EMPTY_ANNOTATIONS = True
cfg.INPUT.MIN_SIZE_TRAIN = (800,)
cfg.INPUT.MIN_SIZE_TEST = 800
cfg.INPUT.MAX_SIZE_TRAIN = 1333
cfg.INPUT.MAX_SIZE_TEST = 1333
# cfg.INPUT.AUGMENTATIONS = augmentation


# Trainer
os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
trainer = DefaultTrainer(cfg)
trainer.resume_or_load(resume=False)
trainer.train()





#---------------------------

# Evaluation
cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")  # Use trained model
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # Confidence threshold
cfg.DATASETS.TEST = (dataset_name,)

evaluator = COCOEvaluator(dataset_name, cfg, False, output_dir="./output/")
val_loader = build_detection_test_loader(cfg, dataset_name)
inference_on_dataset(trainer.model, val_loader, evaluator)
