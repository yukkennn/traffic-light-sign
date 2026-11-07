import json
import numpy as np, pandas as pd, torch
from PIL import Image
from scipy.ndimage import binary_dilation
from tlcv.common import ROOT, CLASSES, load_split
from tlcv.pipeline import Engine, lamp_point, heat_overlay, mask_overlay

torch.set_num_threads(2)
E = Engine(); sam = E.sam()
test = load_split()[2]
models = {"ResNet18 (A1)": "ResNet18 (A1 baseline)", "ResNet18 + CondSim (A2)": "ResNet18 + CondSim (A2)",
          "YOLO11s-cls": "YOLO11s-cls (A2 alternative)"}
rows, examples = [], []
for i, r in test.iterrows():
    crop = Image.open(r.path).convert("RGB")
    big = crop.resize((256, int(256 * crop.size[1] / crop.size[0]))) if crop.size[0] < 256 else crop
    pt = lamp_point(big)
    if pt is None: continue
    res = sam.predict(big, points=[list(pt)], labels=[1], verbose=False, device="cpu")[0].masks
    if res is None or len(res.data) == 0: continue
    lamp = res.data[0].numpy().astype(bool)
    if lamp.mean() > 0.5 or lamp.sum() < 10: continue
    lamp224 = np.asarray(Image.fromarray(lamp.astype(np.uint8) * 255).resize((224, 224), Image.NEAREST)) > 0
    lamp_d = binary_dilation(lamp224, iterations=6)
    row = dict(path=r.path, label=r.label, lamp_frac=float(lamp224.mean()))
    for name, key in models.items():
        p = E.classify(key, [crop])[0]; pred = max(p, key=p.get)
        cam = E.gradcam(key, crop, CLASSES.index(pred))
        y, x = np.unravel_index(np.argmax(cam), cam.shape)
        row[f"{name}|hit"] = bool(lamp_d[y, x]); row[f"{name}|energy"] = float(cam[lamp_d].sum() / (cam.sum() + 1e-9))
        row[f"{name}|correct"] = pred == r.label
        if i % 9 == 0: examples.append((r.path, name, heat_overlay(crop, cam), pred))
    rows.append(row)
    if i % 9 == 0:
        (ROOT / "results" / "xai").mkdir(exist_ok=True)
        mask_overlay(big, np.zeros_like(lamp), lamp, pt).save(ROOT / "results" / "xai" / f"sam_{i}.png")
        for path, name, im, pred in examples[-len(models):]:
            im.save(ROOT / "results" / "xai" / f"cam_{i}_{name.split()[0]}{'_cs' if 'Cond' in name else ''}_{pred}.png")
df = pd.DataFrame(rows); df.to_csv(ROOT / "results" / "xai_per_crop.csv", index=False)
summ = {name: dict(n=len(df), pointing_game=df[f"{name}|hit"].mean(), lamp_energy=df[f"{name}|energy"].mean(),
                   mean_lamp_area=df.lamp_frac.mean()) for name in models}
json.dump(summ, open(ROOT / "results" / "xai_summary.json", "w"), indent=1)
print(json.dumps(summ, indent=1))
