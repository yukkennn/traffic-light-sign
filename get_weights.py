import urllib.request
from pathlib import Path

URL = "https://github.com/ultralytics/assets/releases/download/v8.3.0/"
FILES = ["yolo11s.pt", "yolo11n.pt", "yolo11m.pt", "yolo11s-cls.pt", "yolo11n-cls.pt", "sam2.1_t.pt"]
out = Path(__file__).resolve().parent / "weights"; out.mkdir(exist_ok=True)
for f in FILES:
    p = out / f
    if p.exists() and p.stat().st_size > 1e6: print("ok  ", f); continue
    print("get ", f, flush=True); urllib.request.urlretrieve(URL + f, p)
print("Done. Faster R-CNN (A1 baseline) weights are downloaded automatically by torchvision on first use.")
