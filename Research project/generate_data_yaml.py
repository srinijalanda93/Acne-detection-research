import yaml
import os

# Define the dataset path (inside YOLO_Dataset)
BASE_DIR = "/Users/srinija/Desktop/Research project/YOLO_Dataset"

# YOLO configuration
data_yaml = {
    "path": BASE_DIR,
    "train": "images/train",
    "val": "images/val",
    "nc": 6,  # Number of classes
    "names": ["whiteheads", "blackheads", "papule", "nodule", "keloid", "pustule"]
}

# Save data.yaml inside YOLO_Dataset
yaml_file_path = os.path.join(BASE_DIR, "data.yaml")
with open(yaml_file_path, "w") as yaml_file:
    yaml.dump(data_yaml, yaml_file, default_flow_style=False)

print(f"data.yaml generated successfully at: {yaml_file_path}")
