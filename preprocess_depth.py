import torch
import cv2
import numpy as np
from torchvision.transforms import Compose, Resize, ToTensor
from PIL import Image
import os

# Load MiDaS depth model
model_type = "DPT_Large"
midas = torch.hub.load("intel-isl/MiDaS", model_type)
midas.eval()

# Define preprocessing function
transforms = Compose([Resize((384, 384)), ToTensor()])

def generate_depth(image_path, output_path):
    """Generate synthetic depth maps using MiDaS."""
    img = cv2.imread(image_path)  # Read image with OpenCV (BGR format)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert to RGB

    # Convert NumPy array to uint8 type (Fixes TypeError)
    img = (img * 255).astype(np.uint8) if img.dtype != np.uint8 else img

    # Convert NumPy array to PIL Image before applying transforms
    img_pil = Image.fromarray(img)

    # Apply transforms (resize, convert to tensor)
    input_tensor = transforms(img_pil).unsqueeze(0)

    with torch.no_grad():
        depth = midas(input_tensor)

    depth_map = depth.squeeze().cpu().numpy()
    depth_map = (depth_map - depth_map.min()) / (depth_map.max() - depth_map.min())  # Normalize

    # Convert to 8-bit grayscale image
    depth_map = (depth_map * 255).astype(np.uint8)

    # Save depth map
    cv2.imwrite(output_path, depth_map)

# Process all images
image_dir = "datasets/images"
output_dir = "datasets/depth_maps"
os.makedirs(output_dir, exist_ok=True)

for filename in os.listdir(image_dir):
    if filename.endswith(".jpg"):
        img_path = os.path.join(image_dir, filename)
        output_path = os.path.join(output_dir, filename)
        generate_depth(img_path, output_path)

print("✅ Depth maps saved in", output_dir)
