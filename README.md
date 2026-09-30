# People Detection & Tracking Model

A real-time people detection and multi-object tracking system powered by YOLOv8 and ByteTrack, developed for smart rescue operations and pedestrian monitoring.

## 🚀 Overview

This repository contains the backend implementation for person detection and tracking:
- **Object Detection**: Custom-trained YOLOv8 model (`best.pt`) optimized for person detection in challenging scenarios (water rescue / emergency response).
- **Multi-Object Tracking**: Integration with ByteTrack (`custom_bytetrack.yaml`) for consistent ID tracking across video frames.
- **Real-Time Webcam Pipeline**: Live video feed processing with detection counters and track ID overlays.

---

## 📁 Repository Structure

```text
backend/
├── main.py                   # Real-time webcam tracking with live counter HUD
├── track.py                  # Live ByteTrack person tracking
├── validate.py               # Model evaluation (mAP, Precision, Recall)
├── merge_classes.py          # Class label unification utility
├── data.yaml                 # YOLO dataset configuration
├── README.roboflow.txt       # Roboflow dataset export details
├── trackers/
│   └── custom_bytetrack.yaml # ByteTrack tracker configuration
├── runs/
│   └── detect/
│       └── water_rescue_person/
│           ├── person_model/
│           └── person_model_gpu-2/ # Trained weights (best.pt) and metrics
├── yolo11n.pt                # Base YOLO11 nano model
├── yolov8n.pt                # Base YOLOv8 nano model
└── .gitignore
```

---

## 🛠️ Installation & Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/atharv-nagare45/peopledetectionmodel.git
   cd peopledetectionmodel
   ```

2. **Install dependencies:**
   ```bash
   pip install ultralytics opencv-python torch torchvision
   ```

---

## 🎯 Usage

### 1. Real-Time Tracking & Detection HUD
Run the main tracking script to open the webcam feed with person counting and ID tracking:
```bash
python main.py
```
*Press `Q` to exit the camera feed.*

### 2. Standalone ByteTrack Tracking
```bash
python track.py
```

### 3. Validate Model Performance
Evaluate model metrics (mAP50, mAP50-95, Precision, Recall):
```bash
python validate.py
```

---

## 📊 Dataset & Training

- **Source**: Roboflow People Detection Dataset (7,261 annotated images).
- **Classes**: Single consolidated class (`person`).
- **Trained Weights**: Located under `runs/detect/water_rescue_person/person_model_gpu-2/weights/best.pt`.
