import json
import os
import matplotlib.pyplot as plt
import numpy as np

# Load training history from JSON
with open("training_history.json", "r") as file:
    data = json.load(file)

#Extract YOLOv8 metrics
yolo_metrics = data.get("YOLOv8", {})
precision = yolo_metrics.get("Precision", 0)
recall = yolo_metrics.get("Recall", 0)
map50 = yolo_metrics.get("mAP@50", 0)
map50_95 = yolo_metrics.get("mAP@50-95", 0)

# Create directory for visualizations
os.makedirs("visualizations", exist_ok=True)

# Precision vs Recall Bar Chart**
plt.figure(figsize=(6, 4))
plt.bar(["Precision", "Recall"], [precision, recall], color=["blue", "orange"])
plt.ylim(0, 1)
plt.ylabel("Score")
plt.title("Precision & Recall")
plt.savefig("visualizations/precision_recall.png")
plt.show()

# mAP Scores Bar Chart**
plt.figure(figsize=(6, 4))
plt.bar(["mAP@50", "mAP@50-95"], [map50, map50_95], color=["green", "red"])
plt.ylim(0, 1)
plt.ylabel("Score")
plt.title("Mean Average Precision (mAP)")
plt.savefig("visualizations/map_scores.png")
plt.show()

print("All visualizations saved in 'visualizations' folder!")
