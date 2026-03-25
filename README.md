# Face Mask Detection System

![Python](https://img.shields.io/badge/python-3.8%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)
![YOLOv8](https://img.shields.io/badge/YOLOv8-ultralytics-purple)
![OpenCV](https://img.shields.io/badge/OpenCV-4.x-brightgreen)

A real-time face mask detection system built with **YOLOv8**, supporting webcam and video file inputs with a high-performance OpenCV pipeline.

---

## 🚀 Demo

![Demo](assets/demo.gif)

---

## ✨ Features

- ✅ Real-time mask detection via webcam or video file
- ✅ FPS optimization with frame skipping
- ✅ Color-coded bounding boxes:
  - 🟢 **Green** → Mask
  - 🔴 **Red** → No Mask
- ✅ Optional output video saving (--save)
- ✅ High-performance OpenCV inference pipeline

---

## 🧠 Model Details

| Property | Value |
|----------|-------|
| Architecture | YOLOv8 |
| Task | Object Detection |
| Classes | `Mask`, `No Mask` |
| Framework | `ultralytics` |

---

## 📁 Project Structure
```
face_mask_detection_yolov8/
│
├── src/
│   ├── track.py          # Main inference + webcam/video pipeline
│   └── detect.py         # Single-image detection utility
│
├── models/
│   └── best.pt           # Trained YOLOv8 weights
│
├── assets/
│   └── demo.gif          # Demo preview
│
├── notebooks/
│   └── training.ipynb    # Model training walkthrough
│
├── requirements.txt
├── .gitignore
└── README.md
```

---

## ⚙️ Installation
```bash
# 1. Clone the repo
git clone https://github.com/komal-sukheja/face_mask_detection_yolov8.git
cd face_mask_detection_yolov8

# 2. Create & activate a virtual environment
python -m venv maskenv
mask_env\Scripts\activate        # Windows
source maskenv/bin/activate     # macOS / Linux

# 3. Install dependencies
pip install -r requirements.txt
```

---

## ▶️ Usage

### Run OpenCV Pipeline (High FPS)
```bash
# Webcam (live detection)
python src/track.py --source webcam

# Video file
python src/track.py --source path/to/video.mp4

# Save annotated output
python src/track.py --source path/to/video.mp4 --save
# Output saved as output_test.mp4
```

---

## 📊 Performance

| Mode | FPS |
|------|-----|
| CPU Inference (OpenCV) | ~16–20 FPS |

---

## Training

The full training pipeline — dataset prep, augmentation, hyperparameters, and evaluation — is documented in:
```
notebooks/training.ipynb
```

---

## 🤝 Contributing

Contributions, issues, and feature requests are welcome!
Feel free to fork the repo and open a pull request.

---

## 📄 License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

---

## 📬 Connect With Me

- 💼 **LinkedIn**: [Komal Sukheja](https://www.linkedin.com/in/komal-sukheja/)
- 📧 **Email**: komalsukheja2001@gmail.com
- 🐙 **GitHub**: [komal-sukheja](https://github.com/komal-sukheja)

---

⭐ If you found this project useful, please consider giving it a star — it helps a lot!