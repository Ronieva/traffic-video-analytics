
# Real-Time Traffic Analytics with Object Detection and Tracking

This project implements a real-time **traffic video analytics pipeline** combining object detection, multi-object tracking and **directional vehicle counting using a virtual line**.  
The system is designed with a strong engineering focus, prioritizing **robustness, performance measurement and clean architecture**, and runs efficiently on **CPU**.

---

## 🚗 Features

- Real-time vehicle detection using **YOLOv8**
- Multi-object tracking with persistent IDs
- Virtual line crossing for **vehicle counting**
- Directional counting (**IN / OUT**)
- Performance monitoring (**FPS & latency**)
- Annotated video export (MP4)
- JSON export for post-processing
- CPU-friendly execution (no GPU required)

---

## 🎥 Demo

![Demo](demo.gif)


---

## 🧠 Pipeline Overview

Video Input
↓
YOLOv8 Object Detection
↓
Multi-Object Tracking
↓
Virtual Line Crossing Logic
↓
Directional Vehicle Counting
↓
Overlay & Video Export

---

## ⚙️ Installation

Create and activate a virtual environment (recommended):

bash
python3 -m venv yoloenv
source yoloenv/bin/activate

Install dependencies:
pip install ultralytics opencv-python numpy tqdm

## ▶️ Usage
#Run on a traffic video and save results
PYTHONPATH=. python3 scripts/run_video.py \
  --source 39031.avi \
  --model yolov8n.pt \
  --classes 2 3 5 7 \
  --save_video \
  --save_json

#Run with live visualization only
PYTHONPATH=. python3 scripts/run_video.py \
  --source 39031.avi \
  --model yolov8n.pt \
  --classes 2 3 5 7 \
  --show

##📦 Outputs
. outputs/out.json → Per-frame tracking data

##📊 Performance (example)
Model	Resolution	Device	FPS	Latency
YOLOv8n	960×540	CPU	~20	~45 ms
(Measured on a laptop CPU)

##🚦 Vehicle Counting Logic

Vehicles are counted when their centroid crosses a virtual line

Direction is determined based on centroid motion:

IN → top to bottom

OUT → bottom to top

A cooldown mechanism prevents double counting

##🔧 Design Principles

Modular architecture

Clear separation between detection, tracking and analytics

Reproducible and easy-to-extend codebase

Focus on real-world video analytics use cases
