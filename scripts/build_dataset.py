from ultralytics import YOLO
import cv2, os
import numpy as np
from tqdm import tqdm
from pose_features import extract_pose_features

model = YOLO("yolov8s-pose.pt")

def process_videos(folder, label):
    sequences, labels = [], []

    for video in tqdm(os.listdir(folder)):
        cap = cv2.VideoCapture(os.path.join(folder, video))
        buffer = []

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            res = model(frame, conf=0.4)
            if res[0].keypoints is None:
                continue

            kpts = res[0].keypoints.xy
            if len(kpts) == 0:
                continue

            feat = extract_pose_features(kpts[0].cpu().numpy())
            buffer.append(feat)

            if len(buffer) == 30:
                sequences.append(buffer.copy())
                labels.append(label)
                buffer.pop(0)

        cap.release()

    return sequences, labels

X, y = [], []

fseq, flab = process_videos("data/fall", 1)
nseq, nlab = process_videos("data/normal", 0)

X.extend(fseq + nseq)
y.extend(flab + nlab)

np.save("outputs/X.npy", np.array(X))
np.save("outputs/y.npy", np.array(y))

print("Dataset saved:", len(X), "sequences")

