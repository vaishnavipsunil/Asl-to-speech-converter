# 🤟 ASL to Speech Converter

## 📌 Project Overview

This project is a real-time **American Sign Language (ASL) to Speech Conversion System** built using Computer Vision and Deep Learning.

The system detects ASL hand signs using high-confidence hand landmark detection, predicts letters using a trained deep learning model, appends them to form words, and converts the final word into speech.

---

## 🚀 How It Works

1. Hand landmarks are detected using MediaPipe.
2. Landmark coordinates are passed to a trained deep learning model.
3. The model predicts the ASL letter.
4. Only high-confidence predictions are accepted.
5. Letters are appended to form a word.
6. The word can be edited or spoken using keyboard controls.

---

## 🎮 Keyboard Controls

- **C key** → Remove the last letter  
- **X key** → Clear the entire word  
- **S key** → Speak the word (Text-to-Speech)  
- **Q key** → Quit the application  

---

## 🛠️ Technologies Used

- Python  
- OpenCV  
- MediaPipe (Hand Landmark Detection)  
- TensorFlow / Keras  
- NumPy  
- pyttsx3  

---

## 📂 Project Files

- `collect_full_asl.py` → Script to collect and create ASL landmark dataset  
- `sign_landmarks.csv` → Custom dataset containing hand landmark coordinates  
- `asl_model.h5` → Trained deep learning model  
- `label_encoder.pkl` → Encoded label classes  
- `hand_test.py` → Script to test hand landmark detection  
- `predict_asl.py` → Main prediction and speech conversion script  

---

## 📊 Dataset

- Custom dataset created manually.
- Hand landmark coordinates extracted using MediaPipe.
- Dataset used to train classification model.

---
