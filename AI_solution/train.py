from ultralytics import YOLO
import torch
import torchvision

model = YOLO("../AI_solution/yolov8n.pt")

model.train(
    data="cross.yaml",
    epochs=20,
    imgsz=256,
    batch=4,
    workers=0,
    device="cpu"   # important!
)

print("Training complete! Check runs/detect/train/weights/best.pt")

