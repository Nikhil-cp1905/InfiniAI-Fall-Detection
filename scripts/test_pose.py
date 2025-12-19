from ultralytics import YOLO
import cv2

model = YOLO("yolov8s-pose.pt")

cap = cv2.VideoCapture("data/fall/fall-03-cam0-rgb_start0048.mp4")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame, conf=0.4)
    frame = results[0].plot()

    cv2.imshow("Pose Check", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()

