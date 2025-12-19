
# 🚨 AI-Based Fall Detection System (YOLOv8-Pose + LSTM)

This project implements a **real-time fall detection system** using computer vision and deep learning.  
It detects human falls from videos or live camera feeds, estimates fall velocity, saves evidence, and triggers a **professional web-based alert dashboard**.

The system is designed to be **accurate, explainable, and production-ready**, suitable for healthcare, elderly monitoring, and CCTV-based safety systems.

---

## 🧠 How the System Works (High Level)

```

Video / Camera Input
↓
YOLOv8-Pose (Human Keypoints)
↓
Pose Feature Extraction
↓
Temporal Model (LSTM)
↓
Fall Detected?
↓
Velocity Estimation (m/s)
↓
Frame Capture + Event Logging
↓
Web Alert Dashboard

```

### Key Ideas
- Falls are **temporal events**, not single-frame events
- Human **pose (skeleton)** is more reliable than bounding boxes
- A **temporal LSTM model** distinguishes falls from normal actions
- Physics-based velocity estimation reduces false positives

---

## ✨ Features

- ✅ Real-time fall detection from video or webcam
- 🧍 YOLOv8-based human pose estimation
- ⏱️ Temporal fall classification using LSTM
- 📉 Real-world fall velocity estimation (m/s)
- 🧊 Freeze-frame on fall detection
- 📸 Automatic evidence capture
- 🌐 Professional web alert dashboard
- 📝 Persistent event logging (JSON + CSV)

---

## 📁 Project Structure

```

yolov8project/
├── scripts/
│   ├── detect_fall_video.py      # Inference (main entry point)
│   ├── train_lstm.py             # LSTM training (optional)
│   ├── build_dataset.py          # Dataset builder (optional)
│   └── pose_features.py          # Pose feature extraction
│
├── models/
│   └── fall_lstm.pt              # Trained LSTM weights
│
├── web/
│   ├── app.py                    # Flask web server
│   ├── templates/
│   │   └── alert.html            # Alert dashboard UI
│   └── static/
│       └── images/
│           └── fall.jpg          # Saved fall frame
│
│
├── data/                         # Input videos (optional)
├── yolov8s-pose.pt               # Pretrained YOLOv8 pose weights
├── requirements.txt
└── README.md

````

---

## 🖥️ System Requirements

- Python **3.9+**
- Linux / macOS / Windows
- Optional: NVIDIA GPU (CUDA) for faster inference

---

## 📦 Installation (Step-by-Step)

### 1️⃣ Clone the repository
```bash
git clone <your-repo-url>
cd yolov8project
````

### 2️⃣ Create a virtual environment (recommended)

```bash
python3 -m venv venv
source venv/bin/activate
```

### 3️⃣ Install dependencies

```bash
pip install -r requirements.txt
```

---

## 📥 Model Weights

You need **two weight files**:

### 1. YOLOv8 Pose (pretrained)

Download from Ultralytics:

```bash
https://github.com/ultralytics/assets/releases
```

Place it in project root as:

```
yolov8s-pose.pt
```

### 2. LSTM Fall Model

Place your trained model here:

```
models/fall_lstm.pt
```

>  Without these weights, inference will not work.

---

##  Running the Project (Inference)

### Terminal 1 — Start the Web Alert Server

```bash
python web/app.py
```

You should see:

```
Running on http://127.0.0.1:5000
```

---

### Terminal 2 — Run Fall Detection

```bash
python scripts/detect_fall_video.py
```

By default, it runs on a video file.
To use a webcam, edit `VIDEO_SOURCE` inside `detect_fall_video.py`:

```python
VIDEO_SOURCE = 0
```

---

##  What Happens When a Fall Is Detected

* The video **freezes** on the fall frame
* Fall velocity (m/s) is computed
* Frame is saved for evidence
* Event is logged with timestamp & severity
* Browser opens the **alert dashboard**
* User can press:

  * `r` → resume detection
  * `q` → quit

---

##  Alert Dashboard

The web UI shows:

* Captured fall image
* Fall velocity (m/s)
* Severity level (LOW / MODERATE / HIGH)
* Timestamp
* System status

Accessible at:

```
http://127.0.0.1:5000
```



## ⚙️ Optional: Training the LSTM (Advanced)

If you want to retrain the model:

```bash
python scripts/build_dataset.py
python scripts/train_lstm.py
```

This step is **not required** for inference.

---

## ❓ Common Issues

### ❌ Webcam not opening

Try changing:

```python
VIDEO_SOURCE = 0
```

to:

```python
VIDEO_SOURCE = 1
```

### ❌ Browser does not open

Manually visit:

```
http://127.0.0.1:5000
```

### ❌ Slow performance

* Use a GPU if available
* Reduce video resolution
* Lower YOLO confidence threshold

---

## 📌 Technologies Used

* YOLOv8 (Pose Estimation)
* PyTorch (LSTM)
* OpenCV
* Flask
* NumPy

---

## 📜 License

This project is for **research and educational purposes**.
For medical or commercial deployment, proper validation and certification are required.

---

## 🙌 Acknowledgements

* Ultralytics YOLOv8
* PyTorch Community

---

## 📬 Contact

For questions, improvements, or collaboration, feel free to reach out.

```

---

If you want next, I can:
- shorten this for GitHub showcase
- add diagrams
- add demo GIF instructions
- make a **one-page README** for Slack/demo

Just tell me 👍
```
