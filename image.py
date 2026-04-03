import cv2
import os
import numpy as np
from ultralytics import YOLO
from ppocr_onnx.pipeline import DetAndRecONNXPipeline

# Load the YOLO models
yolo_LP_detect = YOLO(r'model\LP_detector_retrained.pt')
#yolo_LP_detect = YOLO(r'model\plate_yolov8n_320_2024.pt')# License plate detection model
yolo_vehicle_detect = YOLO(r'model\vehicle_yolov8s_640.pt')  # Vehicle detection model

# Initialize the OCR pipeline
ocr_pipeline = DetAndRecONNXPipeline(
    text_det_onnx_model=r'model\ppocrv4\ch_PP-OCRv4_det_infer.onnx',
    text_rec_onnx_model=r'model\ppocrv4\ch_PP-OCRv4_rec_infer.onnx'
)

# Select an image file from the directory
def select_image_file(image_dir):
    images = os.listdir(image_dir)
    if not images:
        print("No images found in the directory.")
        return None
    print("Available images:")
    for idx, image_name in enumerate(images):
        print(f"{idx}: {image_name}")
    try:
        index = int(input("Please select the image by entering the index: "))
        if 0 <= index < len(images):
            return os.path.join(image_dir, images[index])
        else:
            print("Invalid index selected.")
            return None
    except ValueError:
        print("Invalid input. Please enter a valid index.")
        return None

# Save the processed image
def save_processed_image(image, output_dir, image_name):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    output_path = os.path.join(output_dir, image_name)
    cv2.imwrite(output_path, image)
    print(f"Processed image saved to: {output_path}")

# Directory for test images
image_dir = r'test_image'
image_path = select_image_file(image_dir)
if not image_path:
    print("No image file selected.")
    exit()

# Load the selected image
frame = cv2.imread(image_path)
if frame is None:
    print("Failed to load the image.")
    exit()

# Initialize counters
total_plates = 0

# Adjust the confidence threshold for vehicle detection
confidence_threshold = 0.5  # Adjust this value as needed

# Step 1: Detect vehicles with YOLO (Vehicle Detection)
vehicle_results = yolo_vehicle_detect.predict(frame)

# Process detected vehicles with specific colors based on category
for vehicle_result in yolo_vehicle_detect.predict(frame, conf=confidence_threshold):
    vehicle_boxes = vehicle_result.boxes.xyxy.cpu().numpy()
    vehicle_classes = vehicle_result.boxes.cls.cpu().numpy()
    for vehicle_box, vehicle_class in zip(vehicle_boxes, vehicle_classes):
        x, y, xmax, ymax = map(int, vehicle_box[:4])
        class_name = yolo_vehicle_detect.names[int(vehicle_class)]

        # Determine the color based on the category
        if class_name == 'car':
            color = (255, 0, 0)  # Blue
        elif class_name == 'truck':
            color = (255, 0, 255)  # Purple
        elif class_name == 'motorcycle':
            color = (0, 0, 255)  # Red
        elif class_name == 'bus':
            color = (0, 165, 255)  # Orange
        else:
            color = (0, 0, 255)  # Default to red if category is unknown

        # Draw a rectangle around the detected vehicle with the determined color
        cv2.rectangle(frame, (x, y), (xmax, ymax), color, 2)

        # Draw the class name above the rectangle
        if class_name == 'truck':
            cv2.putText(frame, class_name, (xmax - 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        else:
            cv2.putText(frame, class_name, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

# Step 2: Detect license plates with YOLO (License Plate Detection)
lp_results = yolo_LP_detect.predict(frame)

# Process detected license plates
for lp_result in lp_results:
    lp_boxes = lp_result.boxes.xyxy.cpu().numpy()
    total_plates += len(lp_boxes)
    for lp_box in lp_boxes:
        x, y, xmax, ymax = map(int, lp_box[:4])

        # Draw a rectangle around the detected license plate
        cv2.rectangle(frame, (x, y), (xmax, ymax), (0, 255, 0), 2)  # Green box for license plate

        # Crop the license plate region
        lp_region = frame[y:ymax, x:xmax]

        # Use the OCR pipeline to detect and recognize text
        ocr_results = ocr_pipeline.detect_and_ocr(lp_region)

        # Merge all recognized text into one line
        merged_text = ' '.join([ocr_result.text for ocr_result in ocr_results])

        # Draw the merged text above the license plate with a black border
        text_size, _ = cv2.getTextSize(merged_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)  # Increased font size
        text_w, text_h = text_size

        # Draw the text with a black border
        cv2.putText(frame, merged_text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 4)  # Black border
        cv2.putText(frame, merged_text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)  # Green text

# Display total plate count on the top left of the screen
cv2.putText(frame, f'Plates: {total_plates}', (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)


# Save the processed image
output_dir = r'result'
image_name = os.path.basename(image_path)
save_processed_image(frame, output_dir, image_name)

# Show the image with the bounding boxes and recognized text
cv2.imshow("Detected License Plate and Vehicle", frame)
cv2.waitKey(0)
cv2.destroyAllWindows()