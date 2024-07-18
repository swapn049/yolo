import cv2
import numpy as np
import torch
from ultralytics import YOLO



# Define paths
model_path = '/Users/skshah/yolo/runs/detect/train16/weights/best.pt'
image_path = '/Users/skshah/photo1.jpg'

# Load the image
img = cv2.imread(image_path)
H, W, _ = img.shape

# Load the YOLOv8n model
model = YOLO(model_path)
# print("IMAGE : :", img.shape)
results = model.predict(img)
# print("BOXES : : :", results)
# Iterate through the results and draw bounding boxes
for result in results:
    boxes = result.boxes
    for box in boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist()) 
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        label = f"{int(box.cls)} {box.conf.item():.2f}" 
        print("LABEL", label)
        cv2.putText(img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

# Display the image with bounding boxes

cv2.imshow('YOLOv8n Detection', img)
cv2.waitKey(0)
cv2.destroyAllWindows()

