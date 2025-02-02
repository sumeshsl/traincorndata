import json

# Load the COCO JSON file
json_path = "labels_section1annotations.json"
with open(json_path, "r") as f:
    coco_data = json.load(f)

# Extract images and annotations
images = {img["id"]: img for img in coco_data["images"]}
annotations = coco_data["annotations"]

print(f"Total Images: {len(images)}")
print(f"Total Annotations: {len(annotations)}")


import numpy as np

# Camera intrinsic matrix (fx, fy: focal length, cx, cy: principal point)
K = np.array([[1000, 0, 200],  # Adjust focal length as per real-world setup
              [0, 1000, 500],
              [0, 0, 1]])

# Camera extrinsic matrix (identity rotation and translation)
R = np.eye(3)
T = np.array([[0], [0], [-500]])  # Assume the camera is 500mm away

def project_to_3D(segmentation, depth=500):
    """
    Convert 2D segmentation points to 3D using a simple camera model.
    """
    points_2D = np.array(segmentation).reshape(-1, 2)  # Shape (N, 2)
    points_3D = []

    for point in points_2D:
        x, y = point
        point_homogeneous = np.array([x, y, 1])  # Convert to homogeneous coordinates
        point_3D = np.linalg.inv(K) @ (R @ point_homogeneous + T)  # Apply projection
        point_3D *= depth  # Scale using depth
        points_3D.append(point_3D)

    return np.array(points_3D)

# Test projection with one annotation
first_anno = annotations[0]
segmentation = first_anno["segmentation"]
points_3D = project_to_3D(segmentation)

print("Projected 3D points (sample):", points_3D)


import open3d as o3d

all_points_3D = []

# Iterate over all annotations and convert to 3D
for anno in annotations:
    segmentation = anno["segmentation"]
    points_3D = project_to_3D(segmentation)
    all_points_3D.extend(points_3D)  # Collect all points

# Convert to numpy array
all_points_3D = np.array(all_points_3D)

# Check the shape to ensure all points are collected
print(f"Total 3D points: {all_points_3D.shape}")

# Reshape the array to (N, 3)
all_points_3D = all_points_3D.reshape(-1, 3)
# Ensure correct format
all_points_3D = all_points_3D.astype(np.float64)

# Create Open3D PointCloud object
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(all_points_3D)

# Save to PLY file
o3d.io.write_point_cloud("complete_corn_kernels_3D.ply", pcd)
print("Complete 3D point cloud saved successfully!")



# Load the generated 3D point cloud
pcd = o3d.io.read_point_cloud("complete_corn_kernels_3D.ply")

# Convert to a voxel grid (set voxel size based on object scale)
voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(pcd, voxel_size=0.005)

# Save voxelized data
o3d.io.write_voxel_grid("corn_kernels_voxel.ply", voxel_grid)
print("✅ Voxelized 3D data saved!")



# Load the generated 3D point cloud
pcd = o3d.io.read_point_cloud("complete_corn_kernels_3D.ply")

# Convert to NumPy array
all_points_3D = np.asarray(pcd.points)

print(f"Loaded 3D Point Cloud with {all_points_3D.shape[0]} points")

# Prepare COCO-3D format
coco_3d = {
    "images": [{"id": 1, "file_name": "complete_corn_kernels_3D.ply", "width": 512, "height": 512, "depth": 512}],
    "annotations": [],
    "categories": [{"id": 1, "name": "corn kernel"}]
}

# Generate Annotations
annotation_id = 1
for i in range(0, len(all_points_3D), 4):  # Each 4 points define a kernel
    segment = all_points_3D[i:i+4]

    if len(segment) < 4:  # Ignore incomplete segments
        continue

    # Compute 3D bounding box
    x_min, y_min, z_min = segment.min(axis=0)
    x_max, y_max, z_max = segment.max(axis=0)
    bbox = [x_min, y_min, z_min, x_max - x_min, y_max - y_min, z_max - z_min]

    # Create annotation entry
    annotation = {
        "id": annotation_id,
        "image_id": 1,
        "category_id": 1,
        "segmentation": segment.tolist(),
        "bbox": bbox,
        "volume": (bbox[3] * bbox[4] * bbox[5])  # Compute the kernel's volume
    }

    coco_3d["annotations"].append(annotation)
    annotation_id += 1

# Save JSON File
json_path = "corn_kernels_coco3D.json"
with open(json_path, "w") as f:
    json.dump(coco_3d, f, indent=4)

print(f"✅ COCO-3D JSON saved at {json_path}")


with open("corn_kernels_coco3D.json", "r") as f:
    coco_3d = json.load(f)

print(f"Total annotations: {len(coco_3d['annotations'])}")
print("Example Annotation:", coco_3d["annotations"][0])


import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")

# Plot points
for anno in coco_3d["annotations"]:
    segment = np.array(anno["segmentation"])
    ax.scatter(segment[:, 0], segment[:, 1], segment[:, 2])

plt.show()

import json
import numpy as np

json_path = "corn_kernels_coco3D.json"

# ✅ Load the JSON file
with open(json_path, "r") as f:
    coco_data = json.load(f)

# ✅ Find the min/max values across all coordinates for normalization
all_coords = []
for anno in coco_data["annotations"]:
    all_coords.extend(anno["segmentation"])  # Collect all 3D points

all_coords = np.array(all_coords).reshape(-1, 3)  # Convert to numpy array
x_min, y_min, z_min = all_coords.min(axis=0)
x_max, y_max, z_max = all_coords.max(axis=0)

# ✅ Normalize each annotation
for anno in coco_data["annotations"]:
    anno["segmentation"] = [[(x - x_min) / (x_max - x_min),
                              (y - y_min) / (y_max - y_min),
                              (z - z_min) / (z_max - z_min)] for x, y, z in anno["segmentation"]]

    x, y, z, w, h, d = anno["bbox"]
    anno["bbox"] = [(x - x_min) / (x_max - x_min),
                    (y - y_min) / (y_max - y_min),
                    (z - z_min) / (z_max - z_min),
                    w / (x_max - x_min),
                    h / (y_max - y_min),
                    d / (z_max - z_min)]  # Normalize bbox size

    anno["volume"] = anno["volume"] / ((x_max - x_min) * (y_max - y_min) * (z_max - z_min))  # Normalize volume

# ✅ Save the normalized JSON
normalized_json_path = "corn_kernels_coco3D_normalized.json"
with open(normalized_json_path, "w") as f:
    json.dump(coco_data, f, indent=4)

print(f"✅ Normalized dataset saved as {normalized_json_path}")

import os

from detectron2.data.datasets import register_coco_instances
import os

data_dir = os.getcwd()  # Use current directory

# ✅ Register Training Dataset with Normalized JSON
register_coco_instances("corn_kernels_train", {}, f"{data_dir}/corn_kernels_coco3D_normalized.json", data_dir)

# ✅ Register Validation Dataset
register_coco_instances("corn_kernels_val", {}, f"{data_dir}/corn_kernels_coco3D_normalized.json", data_dir)

print("✅ Datasets registered successfully!")



with open("corn_kernels_coco3D_normalized.json", "r") as f:
    coco_data = json.load(f)

print(f"✅ Loaded {len(coco_data['annotations'])} annotations from the normalized dataset")
print("Example Annotation:", coco_data["annotations"][0])



from detectron2.config import get_cfg
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.data import build_detection_test_loader
from detectron2.engine import DefaultTrainer
from detectron2.data import DatasetCatalog, MetadataCatalog

# ✅ Check registered datasets
print("Available datasets:", DatasetCatalog.list())

# ✅ Load dataset dictionary
dataset_dicts = DatasetCatalog.get("corn_kernels_train")
if not dataset_dicts:
    raise ValueError("🚨 No valid data found in 'corn_kernels_train'")

# ✅ Print a sample annotation
print(f"✅ Loaded {len(dataset_dicts)} samples.")
print("Example Sample:", dataset_dicts[0])

# ✅ Load configuration
cfg = get_cfg()
cfg.merge_from_file("configs/corn_kernels_3D.yaml")  # Load the config file used for training
cfg.MODEL.WEIGHTS = "output/model_final.pth"  # Load trained weights

# ✅ Set up evaluator
evaluator = COCOEvaluator("corn_kernels_val", cfg, False, output_dir="./output/")
val_loader = build_detection_test_loader(cfg, "corn_kernels_val")

# ✅ Run inference and evaluation
trainer = DefaultTrainer(cfg)  # Define trainer to use model
inference_on_dataset(trainer.model, val_loader, evaluator)