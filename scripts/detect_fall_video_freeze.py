import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import cv2
import torch
import numpy as np
from ultralytics import YOLO

from scripts.pose_features import extract_pose_features
from scripts.train_lstm import FallLSTM


# ================= CONFIG =================
VIDEO_SOURCE = "data/fall/video7.mp4"   # use 0 for webcam
SEQUENCE_LEN = 30
CONF_THRESHOLD = 0.4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# =========================================


def main():
    pose_model = YOLO("yolov8s-pose.pt")

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
    fall_detected = False

    while True:
        # ---------------- FREEZE MODE ----------------
        if fall_detected:
            cv2.imshow("Fall Detection (YOLOv8-Pose + LSTM)", frozen_frame)
            key = cv2.waitKey(0) & 0xFF

            if key == ord("q"):  # quit
                break
            elif key == ord("r"):  # resume
                fall_detected = False
                feature_buffer.clear()
                continue
            else:
                continue

        # ---------------- NORMAL MODE ----------------
        ret, frame = cap.read()
        if not ret:
            break

        results = pose_model(frame, conf=CONF_THRESHOLD, verbose=False)
        annotated_frame = results[0].plot()

        if results[0].keypoints is not None:
            kpts = results[0].keypoints.xy
            if len(kpts) > 0:
                features = extract_pose_features(
                    kpts[0].cpu().numpy()
                )
                feature_buffer.append(features)

        if len(feature_buffer) > SEQUENCE_LEN:
            feature_buffer.pop(0)

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
                    1.4,
                    (0, 0, 255),
                    4
                )

                frozen_frame = annotated_frame.copy()
                fall_detected = True
                continue

        cv2.imshow("Fall Detection (YOLOv8-Pose + LSTM)", annotated_frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

