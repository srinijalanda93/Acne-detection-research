from ultralytics import YOLO
from history_logger import save_history

# Load trained YOLO model
yolo_model = YOLO("runs/detect/train/weights/best.pt")  # Update the path if necessary

# Validate the model and get results
metrics = yolo_model.val()

# Extracting the relevant metrics correctly
yolo_history = {
    "Precision": metrics.box.p.mean(),  # Mean Precision
    "Recall": metrics.box.r.mean(),  # Mean Recall
    "mAP@50": metrics.box.map50,  # Mean AP at IoU 0.5
    "mAP@50-95": metrics.box.map,  # Mean AP across IoU thresholds (0.5-0.95)
}

# Save history
save_history("YOLOv8", yolo_history)
print("Training history saved successfully!")
