import cv2
import torch
from ultralytics import YOLO

# Load the YOLOv8 model
model = YOLO('/Users/skshah/yolo/runs/detect/train14/weights/best.pt')

# Open the webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to capture image.")
        break

    # Perform object detection
    results = model(frame)

    # Collect all detections
    detections = []
    for result in results:
        boxes = result.boxes
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0].int().tolist()
            class_id = box.cls[0].int().item()
            confidence = box.conf[0].item()
            detections.append((x1, y1, x2, y2, class_id, confidence))

    # Check if "mask_weared_incorrect" (class index 2) with confidence > 0.25 is present
    mask_weared_incorrect_present = any(d[4] == 2 and d[5] > 0.25 for d in detections)

    # Filter detections
    filtered_detections = []
    for detection in detections:
        x1, y1, x2, y2, class_id, confidence = detection
        if mask_weared_incorrect_present:
            # Only add "mask_weared_incorrect" if it is present and confidence > 0.25
            if class_id == 2 and confidence > 0.25:
                filtered_detections.append(detection)
        else:
            # Add all other detections if "mask_weared_incorrect" is not present
            filtered_detections.append(detection)

    # Draw the filtered detections
    for detection in filtered_detections:
        x1, y1, x2, y2, class_id, confidence = detection
        class_name = model.names[class_id]
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f'{class_name} {confidence:.2f}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)

    # Display the resulting frame
    cv2.imshow('Real-Time Object Detection', frame)

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close windows
cap.release()
cv2.destroyAllWindows()
