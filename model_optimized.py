from ultralytics import YOLO
import os

if __name__ == "__main__":
    os.environ["WANDB_MODE"] = "disabled"  # Disable Weights & Biases logging

    model = YOLO("yolov8m.pt")  # Medium model for better accuracy

    result = model.train(
        data="config.yaml",
        epochs=100,         # Sufficient training time
        patience=20,        # Stops early if no improvement
        imgsz=640,         # Larger images improve accuracy
        batch=16,           # Prevents memory issues
        lr0=0.0025,         # Stable learning rate
        lrf=0.01,           # Decay over epochs
        device="cuda",      # Use GPU
        workers=0,          # Faster data loading
        cache=True,         # Cache images in RAM
        val=True,           # Run validation
        augment=True,       # Enable augmentations
        mosaic=1.0,         # Mosaic augmentation (combines images)
        mixup=0.2,          # MixUp augmentation (blends images)
        fliplr=0.5,         # Horizontal flip
        hsv_h=0.015,        # Hue shift
        hsv_s=0.7,          # Saturation shift
        hsv_v=0.4,          # Brightness shift
        overlap_mask=True,  # Helps with overlapping objects
        dropout=0.1,        # Reduces overfitting
        half=True,          # Use mixed precision
        amp=True,           # Enable Automatic Mixed Precision
        verbose=True,       # Print more details
        save=True,          # Save best model
        project="runs/detect",  # Organize runs
        name="model_optimized"  # Custom run name
    )
