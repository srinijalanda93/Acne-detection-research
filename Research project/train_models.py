from ultralytics import YOLO
import torch
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim

from torch.utils.data import DataLoader
from torchvision.models.detection import fasterrcnn_resnet50_fpn
#from custom_dataset import CustomDataset  # to create a custom dataset class for Faster R-CNN

# Paths
DATA_YAML = "/Users/srinija/Desktop/Research project/YOLO_Dataset/data.yaml"
IMG_SIZE = 640
EPOCHS = 50
BATCH_SIZE = 16
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Train YOLOv8
print("Training YOLOv8...")
yolo_model = YOLO("yolov8n.pt")  # pre-trained model
yolo_model.train(data=DATA_YAML, epochs=EPOCHS, imgsz=IMG_SIZE, batch=BATCH_SIZE, device=DEVICE)
print("YOLOv8 Training Completed!")




# # Train Faster R-CNN
# print("Training Faster R-CNN...")
# faster_rcnn = fasterrcnn_resnet50_fpn(pretrained=True)
# faster_rcnn.to(DEVICE)

# # Define optimizer and loss function
# optimizer = optim.Adam(faster_rcnn.parameters(), lr=0.0001)

# def train_faster_rcnn():
#     dataset = CustomDataset("/Users/srinija/Desktop/Research project/YOLO_Dataset/images/train")
#     dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=lambda x: tuple(zip(*x)))
    
#     for epoch in range(EPOCHS):
#         for images, targets in dataloader:
#             images = [img.to(DEVICE) for img in images]
#             targets = [{k: v.to(DEVICE) for k, v in t.items()} for t in targets]
            
#             optimizer.zero_grad()
#             loss_dict = faster_rcnn(images, targets)
#             loss = sum(loss for loss in loss_dict.values())
#             loss.backward()
#             optimizer.step()
            
#         print(f"Epoch {epoch+1}/{EPOCHS}, Loss: {loss.item()}")

#     torch.save(faster_rcnn.state_dict(), "faster_rcnn.pth")
#     print("Faster R-CNN Training Completed!")

# train_faster_rcnn()
