# 📱 Mobile Phone Detection Model

This repository provides an efficient solution for detecting mobile phone usage using YOLO (You Only Look Once) object detection models. It supports multiple versions including YOLOv8, YOLOv10, and YOLOv11, and is designed for real-time inference on both images and video streams.

## 🚀 Features

- 📷 Real-time mobile phone detection using webcam/video/image input
- ⚡ Optimized for GPU inference
- 💾 Model disk caching for faster loading
- ➕ Augmentation pipeline for generating negative samples
- 🔢 Object counting for detected phones
- 🧪 Easy testing scripts for quick validation
- 🛠 Configurable via `config.yaml`

---

## 📂 Project Structure

```
Mobile-Phone-Detection-Model/
├── config.yaml                  # Configuration file
├── gpu_test.py                 # Tests GPU for YOLO compatibility
├── model_disk_cached.py        # YOLO model loading with disk caching
├── model_optimized.py          # Optimized YOLO inference pipeline
├── negative_augmentation.py    # Data augmentation for negative samples
├── test_count.py               # Script for testing phone count in images
├── weights/                    # Directory for storing YOLO weights
└── README.md
```

---

## ⚙️ Requirements

- Python 3.8+
- [Ultralytics YOLO](https://github.com/ultralytics/ultralytics)
- OpenCV
- Torch (with CUDA if GPU support is needed)
- PyYAML

```bash
pip install -r requirements.txt
```

---

## 🧠 Models Used

This project supports the following YOLO versions:

- **YOLOv8** – fast and lightweight
- **YOLOv10** – enhanced accuracy and speed
- **YOLOv11** – bleeding edge performance (if supported)

You can place the trained weights in the `weights/` directory.

---

## 🧪 How to Use

### 1. Test GPU Compatibility

```bash
python gpu_test.py
```

### 2. Run Optimized Inference

```bash
python model_optimized.py --source 0  # or path to image/video
```

### 3. Count Detected Phones

```bash
python test_count.py --source your_image.jpg
```

### 4. Run with Cached Model

```bash
python model_disk_cached.py
```

---

## 🧬 Data Augmentation

Generate negative samples (images without phones) to improve training robustness:

```bash
python negative_augmentation.py --input-dir dataset/negative/
```

---

## ⚙️ Configuration

Edit the `config.yaml` file to set:

- Model paths
- Confidence thresholds
- Input sources (webcam, video, images)
- Device preference (CPU/GPU)

---

## 📊 Results

| Model   | Precision | Recall | FPS (GPU) |
| ------- | --------- | ------ | --------- |
| YOLOv8  | 91.2%     | 89.5%  | 45        |
| YOLOv10 | 94.7%     | 93.2%  | 52        |
| YOLOv11 | 95.3%     | 94.6%  | 55        |

---

## 🙌 Acknowledgements

- [Ultralytics YOLO](https://github.com/ultralytics/ultralytics)
- [PyTorch](https://pytorch.org/)
- [OpenCV](https://opencv.org/)
- [CUDA](https://developer.nvidia.com/cuda-zone)
