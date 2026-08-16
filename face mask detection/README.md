# 😷 Face Mask Detection System

A real-time computer vision application that detects whether a person is wearing a face mask or not using your webcam. 

This project uses OpenCV's Haar Cascades for face detection and a custom TensorFlow/Keras deep learning model (`face_mask_detector.h5`) to classify whether the detected face has a mask on.

## ✨ Features
- **Real-Time Detection:** Processes live video feed from your webcam.
- **Visual Feedback:** Draws bounding boxes around detected faces (Green for Mask, Red for No Mask).
- **Confidence Bar:** Displays a dynamic confidence bar under the bounding box.
- **Live HUD:** Shows real-time FPS and system status.

## 🛠️ Tech Stack
- Python
- OpenCV (`cv2`)
- TensorFlow & Keras
- NumPy

## 🚀 How to Run

1. Ensure you have Python installed on your system.
2. Install the required dependencies:
   ```bash
   pip install opencv-python numpy tensorflow
   ```
3. Run the application:
   ```bash
   python app.py
   ```
4. Press `q` to quit the webcam feed.

## 📁 File Structure
- `app.py`: The main script that handles the webcam feed, face detection, and classification.
- `face_mask_detector.h5`: The pre-trained Keras model used for predicting mask presence.
