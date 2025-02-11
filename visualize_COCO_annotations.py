import cv2
import torch
import numpy as np
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
import matplotlib.pyplot as plt

# Load trained model
cfg = get_cfg()
cfg.merge_from_file("detectron2/configs/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
cfg.MODEL.WEIGHTS = "output/model_final.pth"
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1
predictor = DefaultPredictor(cfg)

# Load image
image_path = "datasets/images/Set_A-F_0011_section_3.jpg"
img = cv2.imread(image_path)
outputs = predictor(img)

# Extract bounding boxes
instances = outputs["instances"].to("cpu")
boxes = instances.pred_boxes.tensor.numpy()

# Draw bounding boxes
for box in boxes:
    x1, y1, x2, y2 = map(int, box)
    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

# Display image
plt.figure(figsize=(10,10))
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.show()

# Count kernels
print(f"Detected Corn Kernels: {len(boxes)}")
