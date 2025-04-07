import os
import shutil
import random
import yaml

# Define dataset paths
BASE_DIR = "/Users/srinija/Desktop/Research project"
IMAGE_SRC = os.path.join(BASE_DIR, "Acne")  # Folder containing all acne images
LABEL_SRC = os.path.join(BASE_DIR, "labels_raw")  # Raw labels 

# Define YOLO folders created 2 folder(images,labels)
IMAGE_DIR = os.path.join(BASE_DIR, "images")
LABEL_DIR = os.path.join(BASE_DIR, "labels")

# Train-Val Split Ratio
TRAIN_RATIO = 0.8  

# Create folders
for sub in ["train", "val"]:
    os.makedirs(os.path.join(IMAGE_DIR, sub), exist_ok=True)
    os.makedirs(os.path.join(LABEL_DIR, sub), exist_ok=True)

# Get all images
image_files = [f for f in os.listdir(IMAGE_SRC) if f.endswith(('.jpg', '.png', '.jpeg'))]
random.shuffle(image_files)

# Split into train and val
split_idx = int(len(image_files) * TRAIN_RATIO)
train_files = image_files[:split_idx]
val_files = image_files[split_idx:]

# Function to move files
def move_files(files, split):
    for file in files:
        src_img = os.path.join(IMAGE_SRC, file)
        dest_img = os.path.join(IMAGE_DIR, split, file)

        label_file = file.replace(file.split('.')[-1], 'txt')  # Match corresponding .txt label
        src_label = os.path.join(LABEL_SRC, label_file)
        dest_label = os.path.join(LABEL_DIR, split, label_file)

        shutil.copy(src_img, dest_img)  # Copy image
        if os.path.exists(src_label):  # Copy label if exists
            shutil.copy(src_label, dest_label)


print("✅ YOLO dataset prepared successfully!")
