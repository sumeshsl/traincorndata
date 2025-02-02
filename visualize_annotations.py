import matplotlib.pyplot as plt
import cv2
import json
import numpy as np
import open3d as o3d

# Load the COCO JSON file
json_path = "labels_section1annotations.json"
with open(json_path, "r") as f:
    coco_data = json.load(f)

# Extract images and annotations
images = {img["id"]: img for img in coco_data["images"]}
annotations = coco_data["annotations"]

# Print sample data
# print(f"Total Images: {len(images)}")
# print(f"Total Annotations: {len(annotations)}")
def visualize_annotations(image_path, annotations, image_id):
    # Read image
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Get annotations for this image
    annos = [anno for anno in annotations if anno["image_id"] == image_id]

    # Draw segmentation masks
    for anno in annos:
        segmentation = np.array(anno["segmentation"]).reshape(-1, 2)
        cv2.polylines(img, [segmentation.astype(np.int32)], isClosed=True, color=(255, 0, 0), thickness=2)

    # Show image
    plt.figure(figsize=(10, 5))
    plt.imshow(img)
    plt.axis("off")
    plt.show()

# Test with the first image
first_image = images[1]
image_path = first_image["file_name"]
visualize_annotations(image_path, annotations, first_image["id"])

# Define intrinsic matrix (assuming a basic camera model)
K = np.array([[1000, 0, 200],  # Focal length in x
              [0, 1000, 500],  # Focal length in y
              [0, 0, 1]])  # Principal point

# Define extrinsic matrix (camera rotation + translation)
R = np.eye(3)  # Identity matrix (no rotation)
T = np.array([[0], [0], [-500]])  # Camera positioned at -500 mm in z


# Project each 2D annotation point into 3D
def project_to_3D(segmentation):
    points_2D = np.array(segmentation).reshape(-1, 2)
    points_3D = []

    for point in points_2D:
        x, y = point
        point_homogeneous = np.array([x, y, 1])  # Convert to homogeneous coordinates
        point_3D = np.linalg.inv(K) @ (R @ point_homogeneous + T)  # Apply projection
        points_3D.append(point_3D)

    return np.array(points_3D)


# Convert first annotation
first_anno = annotations[0]
points_3D = project_to_3D(first_anno["segmentation"])
print("Projected 3D points:", points_3D)

points_3D = points_3D.reshape(-1, 3)
print("Fixed Shape of points_3D:", points_3D.shape)  # Should be (12, 3) if 4 sets of 3 points

# Create an Open3D point cloud object
print("Shape of points_3D:", points_3D.shape)

# Ensure the data is in float64 format
points_3D = points_3D.astype(np.float64)

# Create Open3D PointCloud object
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(points_3D)

# Save to PLY file
o3d.io.write_point_cloud("corn_kernels_3D.ply", pcd)
print("3D point cloud saved successfully!")

# Save the point cloud for further processing
# o3d.io.write_point_cloud("corn_kernels_3D.ply", pcd)
