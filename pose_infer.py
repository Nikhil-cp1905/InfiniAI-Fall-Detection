from ultralytics import YOLO
import cv2

model = YOLO("yolov8s-pose.pt")

cap = cv2.VideoCapture(0)  # webcam

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame, conf=0.4)

    annotated = results[0].plot()
    cv2.imshow("YOLOv8 Pose", annotated)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()

