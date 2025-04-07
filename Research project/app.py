# import streamlit as st
# import torch
# import cv2
# import numpy as np
# from PIL import Image
# from ultralytics import YOLO

# # Load YOLOv8 Model
# MODEL_PATH = "best.pt"  # Change if your model has a different path
# model = YOLO(MODEL_PATH)

# # Streamlit UI
# st.title("Acne Detection using YOLOv8")
# st.write("Upload an image and detect different acne types!")

# # Upload Image
# uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

# if uploaded_file is not None:
#     # Convert uploaded image to OpenCV format
#     image = Image.open(uploaded_file)
#     img_np = np.array(image)
#     img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)  # Convert RGB to BGR (for OpenCV)

#     # Run YOLOv8 Inference
#     results = model(img_bgr)
    
#     # Draw bounding boxes on image
#     for result in results:
#         for box in result.boxes:
#             x1, y1, x2, y2 = map(int, box.xyxy[0])  # Get bounding box
#             conf = box.conf[0].item()  # Confidence score
#             cls = int(box.cls[0].item())  # Class index
#             label = f"{model.names[cls]}: {conf:.2f}"  # Class label
            
#             # Draw Rectangle & Label
#             cv2.rectangle(img_bgr, (x1, y1), (x2, y2), (0, 255, 0), 2)
#             cv2.putText(img_bgr, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
#     # Convert back to RGB for display
#     img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
#     st.image(img_rgb, caption="Detected Image", use_column_width=True)
    
#     # Save Output Image
#     output_path = "detected_image.jpg"
#     cv2.imwrite(output_path, img_bgr)
#     st.success("✅ Detection Complete! Image saved as 'detected_image.jpg'")

import streamlit as st
from PIL import Image
import os
from ultralytics import YOLO
import tempfile
import cv2
import numpy as np

# Set page config
st.set_page_config(page_title="Acne Detection", layout="centered")

# Load YOLOv8 model (Give full path to your trained model)
MODEL_PATH = "/Users/srinija/Desktop/Research project/runs/detect/train/weights/best.pt"  # <- REPLACE with your actual model path
model = YOLO(MODEL_PATH)

# Streamlit title
st.title("🔬 Real-Time Acne Detection using YOLOv8")

# File uploader
uploaded_file = st.file_uploader("Upload an image of acne-affected skin", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Load image
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Run detection
    with st.spinner("Detecting acne..."):
        # Save temp image
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp:
            image.save(temp.name)
            results = model(temp.name)

        # Annotate and show result
        annotated_frame = results[0].plot()
        st.image(annotated_frame, caption="Detection Result", use_column_width=True)

        # Optional: display detection details
        boxes = results[0].boxes
        st.markdown("### Detection Details")
        for i, box in enumerate(boxes):
            cls_id = int(box.cls[0].item())
            conf = float(box.conf[0].item())
            class_name = model.names[cls_id]
            st.write(f"**{i+1}. Class:** {class_name}, **Confidence:** {conf:.2f}")

