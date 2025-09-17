# Plastic Recognition System

This project implements a simple plastic detection system using deep learning and computer vision techniques. It can detect plastic in images and real-time camera feed using a MobileNetV2-based model.

---

## ✅ Features

- Preprocess raw images to a standardized format
- Train a binary classification model to detect plastic
- Predict plastic presence in a single image
- Real-time plastic detection using webcam

---

## ⚡ Requirements

- Python 3.10
- TensorFlow
- OpenCV
- NumPy

plastic_detection_project/  # Raw image dataset (organized by class)
│
├── dataset/               
│   ├── plastic
│   └── clean
│
├── processed_dataset/      # Automatically generated preprocessed images
    ├── plastic
    └── clean     
├── app.py                  # Main Python script
├── README.md               # Project documentation


