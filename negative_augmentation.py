import imgaug.augmenters as iaa
import cv2
import os
import numpy as np

# Paths
input_folder = r'E:\Work Files\Worker Efficiency & Safety - Project\Model - Phone Detection\dataset\negative'
output_folder = r'E:\Work Files\Worker Efficiency & Safety - Project\Model - Phone Detection\dataset_augmented\negative_augmented'  # Save as negative_1
os.makedirs(output_folder, exist_ok=True)

# Augmentation Pipeline
augmenters = iaa.Sequential([
    iaa.Fliplr(0.5),  # Flip horizontally
    iaa.Affine(rotate=(-20, 20)),  # Rotate randomly between -20° to 20°
    iaa.Multiply((0.8, 1.2)),  # Change brightness
    iaa.GaussianBlur(sigma=(0, 1.0)),  # Apply slight blur
])

# Augment images
for filename in os.listdir(input_folder):
    img_path = os.path.join(input_folder, filename)
    img = cv2.imread(img_path)

    if img is None:
        continue  # Skip invalid images

    for i in range(2):  # Create 2 variations per image
        augmented_img = augmenters.augment_image(img)
        output_path = os.path.join(output_folder, f"{filename.split('.')[0]}_aug{i}.jpg")
        cv2.imwrite(output_path, augmented_img)

print("Negative class augmentation completed and saved in 'dataset/negative_1'!")
