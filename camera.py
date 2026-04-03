import cv2
import os
import time
import torch
from ultralytics import YOLO
from ppocr_onnx.pipeline import DetAndRecONNXPipeline

# Check if CUDA is available and set the device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Load the YOLO models and move them to the GPU
yolo_LP_detect = YOLO(r'model\LP_detector_retrained.pt').to(device)
yolo_vehicle_detect = YOLO(r'model\vehicle_yolov8s_640.pt').to(device)

# Initialize the OCR pipeline
ocr_pipeline = DetAndRecONNXPipeline(
    text_det_onnx_model=r'model\ppocrv4\ch_PP-OCRv4_det_infer.onnx',
    text_rec_onnx_model=r'model\ppocrv4\ch_PP-OCRv4_rec_infer.onnx'
)

# Select a video file
def select_video_file(video_dir):
    videos = os.listdir(video_dir)
    if not videos:
        print("No videos found in the directory.")
        return None
    print("Available videos:")
    for idx, video_name in enumerate(videos):
        print(f"{idx}: {video_name}")
    try:
        index = int(input("Please select the video by entering the index: "))
        if 0 <= index < len(videos):
            return os.path.join(video_dir, videos[index])
        else:
            print("Invalid index selected.")
            return None
    except ValueError:
        print("Invalid input. Please enter a valid index.")
        return None

# Function to save the recognized video
def save_recognized_video(input_video_path, output_video_path, frames, original_fps):
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(output_video_path, fourcc, original_fps, (frames[0].shape[1], frames[0].shape[0]))

    for frame in frames:
        out.write(frame)

    out.release()

# Function to generate a unique output video path
def get_unique_output_path(base_path):
    base_name, ext = os.path.splitext(base_path)
    count = 1
    while os.path.exists(base_path):
        base_path = f'{base_name}_{count}{ext}'
        count += 1
    return base_path

# Directory for test videos
video_dir = r'test_video'
video_path = select_video_file(video_dir)
if not video_path:
    print("No video file selected.")
    exit()

# Open the video file
cap = cv2.VideoCapture(video_path)

# Resize factor (adjust the size of the video)
resize_factor = 0.70

# To control the video playback state
video_paused = False

# Initialize frame counter and timer for FPS calculation
frame_count = 0
start_time = time.time()
frames_to_save = []  # List to hold frames for saving

# Get original FPS
original_fps = cap.get(cv2.CAP_PROP_FPS)

# Frame interval for smoother video
desired_fps = 60
frame_interval = 1.0 / desired_fps

# Frame processing function
def process_frame(frame):
    # Resize the frame
    frame_resized = cv2.resize(frame, (0, 0), fx=resize_factor, fy=resize_factor, interpolation=cv2.INTER_LINEAR)

    # Adjust the confidence threshold for vehicle detection
    confidence_threshold = 0.4

    # Step 1: Detect vehicles with YOLO (Vehicle Detection)
    vehicle_results = yolo_vehicle_detect.predict(frame_resized, conf=confidence_threshold, device=device)

    # Process detected vehicles
    for vehicle_result in vehicle_results:
        vehicle_boxes = vehicle_result.boxes.xyxy.cpu().numpy()
        vehicle_classes = vehicle_result.boxes.cls.cpu().numpy()
        for vehicle_box, vehicle_class in zip(vehicle_boxes, vehicle_classes):
            x, y, xmax, ymax = map(int, vehicle_box[:4])
            class_name = yolo_vehicle_detect.names[int(vehicle_class)]

            # Determine the color based on the category
            color = (255, 0, 0) if class_name == 'car' else (255, 0, 255) if class_name == 'truck' else (0, 0, 255) if class_name == 'motorcycle' else (0, 165, 255) if class_name == 'bus' else (0, 0, 255)

            # Draw a rectangle around the detected vehicle
            cv2.rectangle(frame_resized, (x, y), (xmax, ymax), color, 2)

            # Draw the class name above the rectangle
            cv2.putText(frame_resized, class_name, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # Step 2: Detect license plates with YOLO (License Plate Detection)
    lp_results = yolo_LP_detect.predict(frame_resized, device=device)

    # Process detected license plates
    for lp_result in lp_results:
        lp_boxes = lp_result.boxes.xyxy.cpu().numpy()
        for lp_box in lp_boxes:
            x, y, xmax, ymax = map(int, lp_box[:4])

            # Draw a rectangle around the detected license plate
            cv2.rectangle(frame_resized, (x, y), (xmax, ymax), (0, 255, 0), 2)  # Green box for license plate

            # Crop the license plate region
            lp_region = frame_resized[y:ymax, x:xmax]

            # Use the OCR pipeline to detect and recognize text
            ocr_results = ocr_pipeline.detect_and_ocr(lp_region)

            # Merge all recognized text into one line
            merged_text = ' '.join([ocr_result.text for ocr_result in ocr_results])

            # Draw the merged text above the license plate with a black border
            cv2.putText(frame_resized, merged_text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 4)  # Black border
            cv2.putText(frame_resized, merged_text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)  # Green text

    return frame_resized

while cap.isOpened():
    if not video_paused:
        ret, frame = cap.read()
        if not ret:
            break

        # Process the frame
        frame_resized = process_frame(frame)
        frames_to_save.append(frame_resized)  # Store the processed frame for saving

        # Calculate and display FPS
        frame_count += 1
        elapsed_time = time.time() - start_time
        fps = frame_count / elapsed_time
        frame_height, frame_width = frame_resized.shape[:2]
        cv2.putText(frame_resized, f'FPS: {fps:.2f}', (frame_width - 150, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # Show the resized result
        cv2.imshow('Video Frame', frame_resized)

    # Wait for key events
    key = cv2.waitKey(int(frame_interval * 1000)) & 0xFF

    # Check for the close window event (press 'q' to stop the video)
    if key == ord('q'):
        break

    # Check for spacebar to pause or resume video
    if key == ord(' '):
        video_paused = not video_paused

    # Check for the window close (red X button)
    if cv2.getWindowProperty('Video Frame', cv2.WND_PROP_VISIBLE) < 1:
        break

# Release the video capture
cap.release()

# Save the recognized video with a unique name
output_video_path = r'result\video_saved\recognized_video.avi'
output_video_path = get_unique_output_path(output_video_path)

# Save the recognized video until the point of stopping
save_recognized_video(video_path, output_video_path, frames_to_save, original_fps)

# Close all OpenCV windows
cv2.destroyAllWindows()