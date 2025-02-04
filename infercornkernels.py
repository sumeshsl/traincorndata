import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer

# Load trained model configuration
cfg = get_cfg()
cfg.merge_from_file("configs/mask3d_config.yaml")
cfg.MODEL.WEIGHTS = "output/model_final.pth"
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1
cfg.INPUT.FORMAT = "RGB"  # Ensure model expects RGB input

predictor = DefaultPredictor(cfg)

# Load image
image_path = "datasets/images/Set_A-F_0010_section_1.jpg"
img = cv2.imread(image_path)
if img is None:
    raise FileNotFoundError(f"❌ Error: Image not found at {image_path}")

# Ensure image is RGB
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img_h, img_w, _ = img.shape

# **Resize Image to a Detectron2-Compatible Size**
min_size = 800  # Minimum height/width Detectron2 expects
max_size = 1333  # Max size allowed

scale_factor = min(min_size / min(img_h, img_w), max_size / max(img_h, img_w))
new_h, new_w = int(img_h * scale_factor), int(img_w * scale_factor)
img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

# Convert image to float32
img = img.astype(np.float32)

# **Normalize Image Using Model's Pixel Mean and Std**
pixel_mean = np.array(cfg.MODEL.PIXEL_MEAN).reshape(1, 1, 3)  # Reshape for broadcasting
pixel_std = np.array(cfg.MODEL.PIXEL_STD).reshape(1, 1, 3)
img = (img - pixel_mean) / pixel_std  # Normalize

# ✅ Ensure Image is in `(H, W, C)` Format Before Passing to Predictor
outputs = predictor(img)

# Visualize results
v = Visualizer(img.astype(np.uint8), metadata=None, scale=1.0)
v = v.draw_instance_predictions(outputs["instances"].to("cpu"))

plt.figure(figsize=(10,10))
plt.imshow(v.get_image())
plt.show()
