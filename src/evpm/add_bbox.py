from ultralytics import YOLO
from PIL import Image

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import torch

import glob
from tqdm import tqdm

import cv2

import numpy as np

torch.cuda.is_available()

# Load a COCO-pretrained YOLO11n model
model = YOLO("model/best.pt")

STUDENT_TRAIN_DIR = "/path/to/aihub/dataset/110.수학_과목_자동_풀이_데이터/3.개방데이터/1.데이터/Students/train"

STUDENT_TEST_DIR = "/path/to/aihub/dataset/110.수학_과목_자동_풀이_데이터/3.개방데이터/1.데이터/Students/test"


model.eval()

custom_colors = {
    # 0: (0, 255, 0),   # Green for class 0
    0: (255, 0, 0),   # Red for class 1
    # 0: (0, 0, 255),   # Blue for class 2
}

BATCH_SIZE = 256

img_paths = glob.glob(STUDENT_TRAIN_DIR + "/*.jpeg")
model.eval()
for idx, img_path in enumerate(tqdm(img_paths[::BATCH_SIZE])):
    paths = img_paths[BATCH_SIZE * idx : BATCH_SIZE * (idx + 1)]
    with torch.no_grad():
        results = model.predict(source=paths, conf=0.25, iou=0.45, show=False, verbose=False)
    # Loop over results
    for path, r in zip(paths, results):
        im = r.orig_img
        im = cv2.cvtColor(im, cv2.COLOR_RGB2BGR)
        obb_boxes = r.obb.xyxyxyxy.cpu().numpy()  # shape: (N, 4, 2)
        classes = r.obb.cls.cpu().numpy().astype(int)

        for i, corners in enumerate(obb_boxes):
            cls_id = classes[i]
            color = custom_colors.get(cls_id, (255, 255, 255))  # default white

            # Draw the oriented bounding box
            pts = np.array(corners, dtype=np.int32).reshape((-1, 1, 2))
            cv2.polylines(im, [pts], isClosed=True, color=color, thickness=2)
        # Save the image with bounding boxes
        path = path.replace("Students", "Students_bbox_red")
        cv2.imwrite(path, im)

print(torch.cuda.list_gpu_processes())

BATCH_SIZE = 256

img_paths = glob.glob(STUDENT_TEST_DIR + "/*.jpeg")
faulty_idx = img_paths.index("/path/to/aihub/dataset/110.수학_과목_자동_풀이_데이터/3.개방데이터/1.데이터/Students/test/P4_2_03_24748_75233_19_O.jpeg")
img_paths.pop(faulty_idx)

for idx, img_path in enumerate(tqdm(img_paths[::BATCH_SIZE])):
    paths = img_paths[BATCH_SIZE * idx : BATCH_SIZE * (idx + 1)]
    results = model.predict(source=paths, conf=0.25, iou=0.45, device=0, show=False, verbose=False)
    # Loop over results
    for path, r in zip(paths, results):
        im = r.orig_img
        im = cv2.cvtColor(im, cv2.COLOR_RGB2BGR)
        obb_boxes = r.obb.xyxyxyxy.cpu().numpy()  # shape: (N, 4, 2)
        classes = r.obb.cls.cpu().numpy().astype(int)

        for i, corners in enumerate(obb_boxes):
            cls_id = classes[i]
            color = custom_colors.get(cls_id, (255, 255, 255))  # default white

            # Draw the oriented bounding box
            pts = np.array(corners, dtype=np.int32).reshape((-1, 1, 2))
            cv2.polylines(im, [pts], isClosed=True, color=color, thickness=2)
        # Save the image with bounding boxes
        path = path.replace("Students", "Students_bbox_red")
        cv2.imwrite(path, im)

import numpy as np
for p in img_paths:
    try:
        img = Image.open(p).convert("RGB")
        np.asarray(img).shape
    except:
        print(f"Error opening image: {p}")
        continue




