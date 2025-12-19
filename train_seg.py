from ultralytics import YOLO

model = YOLO("yolov8s-seg.pt")

model.train(
    data="data.yaml",
    epochs=100,
    imgsz=640,
    batch=8,
    device=0,
    workers=8
)

