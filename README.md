# 👁 Face Recognition & Emotion Detection System

A real-time computer vision system built with Python that detects faces, recognizes people by name, and analyzes emotions, age, and gender live from a webcam.

## Features
- 😊 Real-time emotion detection (7 emotions)
- 👤 Face registration and recognition
- 🎂 Age and gender estimation
- 📋 Attendance logging to CSV
- 📧 Email alerts for unknown faces
- 📊 Live web dashboard (Flask)

## Tech Stack
Python, OpenCV, DeepFace, face_recognition, dlib, Flask

## Setup
```bash
conda create -n cv python=3.11 -y
conda activate cv
conda install -c conda-forge dlib -y
pip install opencv-python deepface face_recognition numpy pandas flask
```

## Usage
Terminal 1: `python face_emotion_cv.py`
Terminal 2: `python dashboard.py`
Dashboard: http://127.0.0.1:5000
