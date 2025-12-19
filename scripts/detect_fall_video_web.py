import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import cv2
import torch
import numpy as np
import webbrowser
from ultralytics import YOLO
from datetime import datetime

from scripts.pose_features import extract_pose_features
from scripts.train_lstm import FallLSTM
from web.app import update_fall_data


# ================= CONFIG =================
VIDEO_SOURCE = "data/fall/video10.mp4"   # use 0 for webcam
SEQUENCE_LEN = 30
CONF_THRESHOLD = 0.4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
HUMAN_TORSO_HEIGHT_M = 0.5   # average shoulder–hip distance
ALERT_URL = "http://127.0.0.1:5000"
# =========================================


def main():
    # Load models
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

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 30

    feature_buffer = []
    hip_y_history = []
    fall_detected = False

    while True:
        # ---------------- FREEZE MODE ----------------
        if fall_detected:
            cv2.imshow("Fall Detection", frozen_frame)
            key = cv2.waitKey(0) & 0xFF

            if key == ord("q"):
                break
            elif key == ord("r"):
                fall_detected = False
                feature_buffer.clear()
                hip_y_history.clear()
                continue
            else:
                continue

        # ---------------- NORMAL MODE ----------------
        ret, frame = cap.read()
        if not ret:
            break

        results = pose_model(frame, conf=CONF_THRESHOLD, verbose=False)
        annotated_frame = results[0].plot()

        velocity_mps = 0.0

        if results[0].keypoints is not None:
            kpts = results[0].keypoints.xy
            if len(kpts) > 0:
                kp = kpts[0].cpu().numpy()

                # Key joints
                left_hip, right_hip = kp[11], kp[12]
                left_sh, right_sh = kp[5], kp[6]

                hip_y = (left_hip[1] + right_hip[1]) / 2
                shoulder_y = (left_sh[1] + right_sh[1]) / 2

                torso_px = abs(shoulder_y - hip_y)
                meters_per_pixel = (
                    HUMAN_TORSO_HEIGHT_M / torso_px if torso_px > 1 else 0
                )

                hip_y_history.append(hip_y)
                if len(hip_y_history) > 5:
                    hip_y_history.pop(0)

                if len(hip_y_history) >= 2:
                    v_px = np.mean(np.diff(hip_y_history))
                    velocity_mps = abs(v_px * meters_per_pixel * fps)

                features = extract_pose_features(kp)
                feature_buffer.append(features)

        if len(feature_buffer) > SEQUENCE_LEN:
            feature_buffer.pop(0)

        # Display velocity
        cv2.putText(
            annotated_frame,
            f"Fall Velocity: {velocity_mps:.2f} m/s",
            (30, 100),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0, 255, 255),
            2
        )

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
                # Overlay alert
                cv2.putText(
                    annotated_frame,
                    "FALL DETECTED",
                    (40, 60),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.4,
                    (0, 0, 255),
                    4
                )

                # Save fall frame
                os.makedirs("web/static/images", exist_ok=True)
                cv2.imwrite("web/static/images/fall.jpg", annotated_frame)

                # Update web data
                update_fall_data(round(velocity_mps, 2))

                # Open browser alert
                webbrowser.open(ALERT_URL)

                frozen_frame = annotated_frame.copy()
                fall_detected = True
                continue

        cv2.imshow("Fall Detection", annotated_frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

