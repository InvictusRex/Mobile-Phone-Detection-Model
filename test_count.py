import os
from ultralytics import YOLO

# Load trained YOLO model
model = YOLO(r'E:\Work Files\Project - Worker Efficiency & Safety\Model - Phone Detection\src\runs\detect\model_optimized2\weights\best.pt')

# Path to unseen test images
test_images_path = r'E:\Work Files\Project - Worker Efficiency & Safety\Model - Phone Detection\dataset\test\images'

# Get list of all image files in the folder
image_files = [f for f in os.listdir(test_images_path) if f.endswith(('.jpg', '.png', '.jpeg'))]

# Run inference and count phones per image
for image_file in image_files:
    image_path = os.path.join(test_images_path, image_file)

    # Run inference
    results = model(image_path)

    # Count detected phones
    for result in results:
        num_phones = sum(1 for box in result.boxes if box.conf > 0.5)  # Confidence threshold
        print(f"Image: {image_file} - Phones detected: {num_phones}")
