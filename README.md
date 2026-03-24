# Face Liveness Detection using MobileNetV2

A deep learning based **Face Liveness Detection system** designed to detect spoofing attacks such as printed photos, replay attacks, and mask attacks.
The model is trained using the **ROSE-P3 dataset** and uses **MobileNetV2 transfer learning** for binary classification between **real** and **fake** faces.

This model is designed to be **converted to TensorFlow Lite (TFLite)** and integrated into **Android applications** for real-time liveness detection.

---

## 🚀 Features

* Detects **real vs spoof faces**
* Uses **MobileNetV2 pretrained architecture**
* Data augmentation for better generalization
* Extracts frames automatically from video dataset
* Binary classification using **sigmoid activation**
* Model conversion to **TensorFlow Lite**
* Optimized for **mobile deployment**

---

## 📂 Dataset

This project uses the **ROSE-P3 face anti-spoofing dataset** which contains multiple attack types.

Attack categories used:

* Real face
* Printed mask attacks
* 3D mask attacks
* Screen replay attacks
* Outline attacks

Dataset structure used in this project:

```
dataset/
    rose_p3/
        real/
        mask/
        mask3d/
        monitor/
        outline/
        outline3d/
```

Frames are automatically extracted and stored in:

```
frame_dataset/
    real_z/
    fake/
```

---

## 🧠 Model Architecture

The model uses **transfer learning** with MobileNetV2.

Architecture:

```
MobileNetV2 (ImageNet pretrained)
        ↓
GlobalAveragePooling2D
        ↓
Dense (128, ReLU)
        ↓
Dropout
        ↓
Dense (1, Sigmoid)
```

Output:

```
0 → Fake
1 → Real
```

---

## 🏋️ Training Pipeline

1. Extract frames from videos
2. Generate dataset (real / fake)
3. Apply data augmentation
4. Train MobileNetV2 model
5. Save trained model (.h5)
6. Convert model to TensorFlow Lite

---

## ⚙️ Installation

Clone the repository:

```
git clone https://github.com/sankalp125/face-liveness-detection.git
cd face-liveness-detection
```

Install dependencies:

```
pip install -r requirements.txt
```

---

## ▶️ Training the Model

Run the training script:

```
python train_model.py
```

This will:

* Extract frames from videos
* Train the model
* Save the trained model

Output files:

```
liveness_model.h5
liveness_model.tflite
```

---

## 📱 Android Integration

The trained model can be integrated into an **Android application** using **TensorFlow Lite**.

Typical pipeline:

```
CameraX → Face Detection → Face Crop → TFLite Model → Real / Spoof Prediction
```

---

## 📊 Example Output

The training process generates accuracy graphs for:

* Training accuracy
* Validation accuracy

These graphs help evaluate model performance and detect overfitting.

---

## 🛠 Tech Stack

* Python
* TensorFlow / Keras
* OpenCV
* MediaPipe
* MobileNetV2
* Matplotlib

---

## 🎯 Future Improvements

Possible improvements for higher security:

* Blink detection
* Micro-expression analysis
* Temporal liveness detection
* Multi-frame inference
* Depth estimation

---

## 📜 License

This project is open-source and available under the **MIT License**.

---

## 👨‍💻 Author

**Sankalp Tiwari**

Android Developer | Machine Learning Enthusiast

---
