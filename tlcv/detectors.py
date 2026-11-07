import time
from pathlib import Path
import numpy as np
import torch
from PIL import Image, ImageOps

WEIGHTS = Path(__file__).resolve().parents[1] / "weights"
COCO_TL_YOLO = 9


def load_image(path):
    return ImageOps.exif_transpose(Image.open(path)).convert("RGB")


class FasterRCNN:
    name = "Faster R-CNN MobileNetV3 (A1)"
    def __init__(self, min_size=960, max_size=1600, thr=0.10):
        from torchvision.models.detection import fasterrcnn_mobilenet_v3_large_fpn, FasterRCNN_MobileNet_V3_Large_FPN_Weights
        w = FasterRCNN_MobileNet_V3_Large_FPN_Weights.DEFAULT
        self.tl = w.meta["categories"].index("traffic light")
        self.m = fasterrcnn_mobilenet_v3_large_fpn(weights=w, box_score_thresh=thr).eval()
        self.m.transform.min_size, self.m.transform.max_size = (min_size,), max_size
    @torch.no_grad()
    def __call__(self, img):
        from torchvision.transforms.functional import to_tensor
        o = self.m([to_tensor(img)])[0]
        return [(*b, float(s)) for b, l, s in zip(o["boxes"].tolist(), o["labels"].tolist(), o["scores"].tolist()) if l == self.tl]


class YOLO11:
    def __init__(self, variant="s", imgsz=1280, thr=0.10):
        from ultralytics import YOLO
        self.m = YOLO(str(WEIGHTS / f"yolo11{variant}.pt")); self.imgsz, self.thr = imgsz, thr
        self.name = f"YOLO11{variant} @ {imgsz}"
    def __call__(self, img):
        r = self.m.predict(img, imgsz=self.imgsz, conf=self.thr, classes=[COCO_TL_YOLO], verbose=False, device="cpu")[0]
        return [(*b, float(s)) for b, s in zip(r.boxes.xyxy.tolist(), r.boxes.conf.tolist())]
