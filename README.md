# **Eye Detection Inference**
## **Description** 

This project implements a lightweight system for detecting eyes and determining their state (open or closed) in real time. Designed for deployment on System-on-Chip (SoC) devices such as the Jetson Nano, it utilizes TensorFlow Lite models for efficient inference and OpenCV for video frame processing.

The system comprises:

* **Face Detection** : Identifies the face and crops the region of interest for the eyes.
* **Eye State Detection** : Infers whether the eyes are open or closed using a TensorFlow Lite eye detection model.
* **Features**
- TensorFlow Lite integration for running inference on SoCs.
- Real-time video frame processing using OpenCV.
- Drowsiness detection by monitoring eye states over time.
- Performance metrics, including inference time and frame rate.
