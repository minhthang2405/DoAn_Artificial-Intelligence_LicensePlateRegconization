import os
import torch
import time
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
from ultralytics import YOLO

def main():
    data_yaml = 'dataset.yaml'
    save_path = r'model\LP_detector_retrained.pt'

    # Check if CUDA is available
    if torch.cuda.is_available():
        device = '0'  # Use the first GPU
    else:
        device = 'cpu'  # Fallback to CPU

    # Load a YOLOv8 model
    model = YOLO('yolov8n.pt')  # You can choose different model sizes like yolov8s.pt, yolov8m.pt, etc.

    # Start timing the training process
    start_time = time.time()

    # Train the model
    model.train(data=data_yaml, epochs=50, imgsz=640, device=device)

    # End timing the training process
    end_time = time.time()

    # Calculate the total training time
    total_time = end_time - start_time
    print(f"Total training time: {total_time:.2f} seconds")

    # Save the trained model
    model.save(save_path)  # Save the model as a .pt file

if __name__ == '__main__':
    main()
