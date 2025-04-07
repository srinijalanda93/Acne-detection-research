import torch
import os
import cv2
import numpy as np
from torch.utils.data import Dataset

class CustomDataset(Dataset):
    def __init__(self, image_dir, label_dir, transform=None):
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.transform = transform
        self.image_filenames = os.listdir(image_dir)

    def __getitem__(self, index):
        image_filename = self.image_filenames[index]
        img_path = os.path.join(self.image_dir, image_filename)
        label_path = os.path.join(self.label_dir, image_filename.replace(".jpg", ".txt"))

        # Load image
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = image / 255.0  # Normalize

        # Load labels
        boxes = []
        with open(label_path, "r") as f:
            for line in f.readlines():
                class_id, x, y, w, h = map(float, line.strip().split())
                boxes.append([class_id, x, y, w, h])

        boxes = torch.tensor(boxes, dtype=torch.float32)

        # Apply transformations
        if self.transform:
            image = self.transform(image)

        target = {"boxes": boxes[:, 1:], "labels": boxes[:, 0].long()}

        return image, target

    def __len__(self):
        return len(self.image_filenames)
