import cv2
import numpy as np
import onnxruntime as ort

def preprocess(frame):
    # Convert BGR to RGB if model trained on RGB images
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Resize the frame to model's expected input size
    frame_resized = cv2.resize(frame_rgb, (640, 640))  # Adjust size as needed
    
    # Normalize the frame (ensure dtype is float32)
    frame_normalized = frame_resized.astype(np.float32) / 255.0
    
    # Change data layout from HWC to CHW
    frame_input = np.transpose(frame_normalized, (2, 0, 1))
    
    # Add a batch dimension
    frame_input = np.expand_dims(frame_input, axis=0)
    
    return frame_input

def postprocess(frame, outputs, threshold=0.5):
    # Loop through detections in the outputs
    for detection in outputs[0][0]:  # Adjust based on your model's specific output format
        conf = detection[2]
        if conf >= threshold:
            x, y, w, h = [int(detection[i] * frame.shape[j % 2]) for i, j in zip(range(3, 7), (1, 0, 1, 0))]
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            label = f"Mask Detected: {conf:.2f}"
            cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    return frame

# Main function and webcam setup remains the same
def main():
    # Load ONNX model
    session = ort.InferenceSession("/Users/skshah/Downloads/Model16/weights/best.onnx")
    input_name = session.get_inputs()[0].name

        # Start webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Cannot open camera")
        exit()

    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break

        # Preprocess the frame
        frame_input = preprocess(frame)

        # Compute
        outputs = session.run(None, {input_name: frame_input})

        # Postprocess and display the frame
        frame_display = postprocess(frame, outputs)
        cv2.imshow('Object Detection', frame_display)

        if cv2.waitKey(1) == ord('q'):
            break

    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
