from ultralytics import YOLO
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# Load a pretrained YOLO model (recommended for training)
model = YOLO("yolo11n-obb.pt")

results = model.train(data="./data/dataset.yaml", epochs=200, imgsz=224)

# Evaluate the model's performance on the validation set
results = model.val()

# Export the model to ONNX format
success = model.export(format="onnx")