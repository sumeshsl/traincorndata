import torch
print(torch.__version__)  # Should show a version with 'cu' instead of 'cpu'
print(torch.cuda.is_available())  # Should return True if GPU is available
import json

json_path = "corn_kernels_coco3D.json"

# ✅ Load the JSON file
with open(json_path, "r") as f:
    coco_data = json.load(f)

# ✅ Check if annotations exist
if len(coco_data.get("annotations", [])) == 0:
    raise ValueError(f"🚨 Error: No annotations found in {json_path}")

# ✅ Print some data to verify
print(f"✅ Loaded {len(coco_data['annotations'])} annotations from {json_path}")
print("Example Annotation:", coco_data["annotations"][0])


import os

data_dir = os.getcwd()  # Set this to your dataset directory if different

# ✅ Check JSON file
json_file = os.path.join(data_dir, "corn_kernels_coco3D.json")
if not os.path.exists(json_file):
    raise FileNotFoundError(f"🚨 JSON file missing: {json_file}")

# ✅ Check if .ply files exist
ply_file = os.path.join(data_dir, "complete_corn_kernels_3D.ply")
if not os.path.exists(ply_file):
    raise FileNotFoundError(f"🚨 PLY file missing: {ply_file}")

print("✅ All dataset files are found and ready for training.")
