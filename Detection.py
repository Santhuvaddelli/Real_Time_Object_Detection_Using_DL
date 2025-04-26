import cv2
import numpy as np
import time
from ultralytics import YOLO

# Load YOLOv3 (COCO dataset)
yolo_v3_net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")

# Load COCO class names
with open("coco.names", "r") as f:
    coco_classes = [line.strip() for line in f.readlines()]

yolo_v3_layer_names = yolo_v3_net.getLayerNames()
yolo_v3_output_layers = [yolo_v3_layer_names[i - 1] for i in yolo_v3_net.getUnconnectedOutLayers()]

# Load Custom YOLOv8 Model
yolo_v8_model = YOLO("best.pt")  
yolo_v8_model.overrides['verbose'] = False
custom_classes = yolo_v8_model.names  

# Generate random colors for classes
colors = np.random.uniform(0, 255, size=(max(len(coco_classes), len(custom_classes)), 3))

# Choose input type
input_type = input("Choose input type (image/video/camera): ").strip().lower()
cap = None
frame = None
frame_count = 0
skip_frames = 3  # Adjusted for smoother processing

if input_type == "image":
    image_path = input("Enter the image path: ").strip()
    frame = cv2.imread(image_path)
    if frame is None:
        print("Error: Could not load image.")
        exit()
elif input_type == "video":
    video_path = input("Enter the video path: ").strip()
    cap = cv2.VideoCapture(video_path)
elif input_type == "camera":
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)
else:
    print("Invalid input type. Exiting.")
    exit()

# Function to perform YOLOv3 detection
def detect_objects_v3(frame):
    h, w, _ = frame.shape
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    yolo_v3_net.setInput(blob)
    outs = yolo_v3_net.forward(yolo_v3_output_layers)

    class_ids, confidences, boxes = [], [], []
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                center_x, center_y, width, height = (detection[0] * w, detection[1] * h, detection[2] * w, detection[3] * h)
                x, y = int(center_x - width / 2), int(center_y - height / 2)
                boxes.append([x, y, int(width), int(height)])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
    return boxes, class_ids, indexes, coco_classes

# Function to perform YOLOv8 detection
def detect_objects_v8(frame):
    results = yolo_v8_model(frame)
    boxes, class_ids, confidences = [], [], []

    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            class_id = int(box.cls[0])
            confidence = float(box.conf[0])
            boxes.append([x1, y1, x2 - x1, y2 - y1])
            confidences.append(confidence)
            class_ids.append(class_id)

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.3, 0.4)
    return boxes, class_ids, indexes, custom_classes

# Function to process and display results
def process_frame(frame):
    v3_boxes, v3_class_ids, v3_indexes, v3_classes = detect_objects_v3(frame)
    v8_boxes, v8_class_ids, v8_indexes, v8_classes = detect_objects_v8(frame)

    font = cv2.FONT_HERSHEY_SIMPLEX

    def draw_boxes(boxes, class_ids, indexes, classes):
        for i in range(len(boxes)):
            if i in indexes:
                x, y, w, h = boxes[i]
                label = f"{classes[class_ids[i]]}"
                color = colors[class_ids[i] % len(colors)]
                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                cv2.putText(frame, label, (x , y - 10), font, 1.0 , (0,0,0) , 2)

    draw_boxes(v3_boxes, v3_class_ids, v3_indexes, v3_classes)
    draw_boxes(v8_boxes, v8_class_ids, v8_indexes, v8_classes)

    # Resize for better visualization
    resized_frame = cv2.resize(frame, (1280, 720))
    cv2.imshow("Object Detection", resized_frame)

# Process image
if input_type == "image":
    process_frame(frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Process video
elif input_type == "video":
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        if frame_count % skip_frames == 0:  # Skip frames for smoothness
            process_frame(frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):  # Faster display refresh
            break

    cap.release()
    cv2.destroyAllWindows()

# Process camera
elif input_type == "camera":
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        process_frame(frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):  # Minimal delay
            break

    cap.release()
    cv2.destroyAllWindows()
