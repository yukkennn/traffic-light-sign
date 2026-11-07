import time
import numpy as np, pandas as pd, torch
from tlcv.common import ROOT, DATA
from tlcv.detectors import FasterRCNN, YOLO11, load_image

torch.set_num_threads(max(1, torch.get_num_threads()))
imgs = [load_image(p) for p in sorted((DATA / "images" / "full").glob("*.jpg"))[:5]]
csv = ROOT / "results" / "detector_comparison.csv"; df = pd.read_csv(csv)
for D, key in [(FasterRCNN(480, 800), "Faster R-CNN MobileNetV3 (A1) @ 480"),
               (FasterRCNN(960, 1600), "Faster R-CNN MobileNetV3 (A1) @ 960"),
               (YOLO11("n", 1280), "YOLO11n @ 1280"), (YOLO11("s", 1280), "YOLO11s @ 1280"),
               (YOLO11("m", 1280), "YOLO11m @ 1280"), (YOLO11("s", 1920), "YOLO11s @ 1920")]:
    for _ in range(2): D(imgs[0])
    ts = []
    for im in imgs:
        t0 = time.perf_counter(); D(im); ts.append(time.perf_counter() - t0)
    ms = 1000 * float(np.mean(ts)); df.loc[df.detector == key, ["latency_ms", "FPS"]] = [ms, 1000 / ms]
    df.loc[df.detector == key, "latency_clean"] = True
    print(key, f"{ms:.0f} ms", flush=True)
df.round(4).to_csv(csv, index=False)
