import json
import pandas as pd, torch
from tlcv.common import ROOT, DATA
from tlcv.det_eval import iou
from tlcv.detectors import load_image
from tlcv.pipeline import Engine, pad_box, CLASSIFIERS

torch.set_num_threads(2)
gt = json.load(open(ROOT / "results" / "scene_gt.json"))["images"]
dets = json.load(open(ROOT / "results" / "scene_dets.json"))
E = Engine()
pairs = [("Faster R-CNN MobileNetV3 (A1) @ 960", "ResNet18 (A1 baseline)"),
         ("YOLO11s @ 1280", "ResNet18 (A1 baseline)"),
         ("YOLO11s @ 1280", "YOLO11s-cls (A2 alternative)"),
         ("YOLO11s @ 1280", "ResNet18 + CondSim (A2)")]
rows = []
for dk, ck in pairs:
    n = solved = matched = correct = 0
    for k, g in gt.items():
        im = load_image(DATA / "images" / "full" / k); W, H = im.size
        D = [d for d in dets[dk][k] if d[4] >= 0.30]
        used = set()
        for light in g["lights"]:
            n += 1
            best = max(((iou(d, light["box"]), j) for j, d in enumerate(D) if j not in used), default=(0, -1))
            if best[0] < 0.5: continue
            used.add(best[1]); matched += 1
            p = E.classify(ck, [im.crop(pad_box(D[best[1]], W, H))])[0]
            if max(p, key=p.get) == light["state"]: correct += 1; solved += 1
    rows.append(dict(detector=dk, classifier=ck, n_lights=n, detected=matched, state_acc_on_detected=correct / max(matched, 1),
                     end_to_end_recall=solved / n))
    print(rows[-1], flush=True)
pd.DataFrame(rows).round(4).to_csv(ROOT / "results" / "end_to_end.csv", index=False)
