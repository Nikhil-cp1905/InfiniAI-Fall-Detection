import sys
import os

# Allow imports from project root
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import cv2
import torch
import numpy as np
from ultralytics import YOLO

from scripts.pose_features import extract_pose_features
from scripts.train_lstm import FallLSTM


# ================= CONFIG =================
VIDEO_SOURCE = "data/fall/video1.mp4"   # use 0 for webcam
SEQUENCE_LEN = 30
CONF_THRESHOLD = 0.4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# =========================================


def main():
    # Load YOLOv8 Pose model
    pose_model = YOLO("yolov8s-pose.pt")

    # Load trained LSTM model
    lstm_model = FallLSTM().to(DEVICE)
    lstm_model.load_state_dict(
        torch.load("models/fall_lstm.pt", map_location=DEVICE)
    )
    lstm_model.eval()

    cap = cv2.VideoCapture(VIDEO_SOURCE)
    if not cap.isOpened():
        print("❌ Could not open video source")
        return

    feature_buffer = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # ---------------- POSE INFERENCE ----------------
        results = pose_model(frame, conf=CONF_THRESHOLD, verbose=False)

        # THIS DRAWS THE SKELETON (IMPORTANT)
        annotated_frame = results[0].plot()

        # ---------------- FEATURE EXTRACTION ----------------
        if results[0].keypoints is not None:
            kpts = results[0].keypoints.xy
            if len(kpts) > 0:
                features = extract_pose_features(
                    kpts[0].cpu().numpy()
                )
                feature_buffer.append(features)

        # Maintain fixed-length buffer
        if len(feature_buffer) > SEQUENCE_LEN:
            feature_buffer.pop(0)

        # ---------------- FALL PREDICTION ----------------
        if len(feature_buffer) == SEQUENCE_LEN:
            seq = torch.tensor(
                [feature_buffer],
                dtype=torch.float32,
                device=DEVICE
            )

            with torch.no_grad():
                logits = lstm_model(seq)
                pred = torch.argmax(logits, dim=1).item()

            if pred == 1:
                cv2.putText(
                    annotated_frame,
                    "FALL DETECTED",
                    (40, 60),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.3,
                    (0, 0, 255),
                    3
                )

        # ---------------- DISPLAY ----------------
        cv2.imshow("Fall Detection (YOLOv8-Pose + LSTM)", annotated_frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

