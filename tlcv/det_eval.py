import json, time
import numpy as np, pandas as pd, torch
from tlcv.common import ROOT, PHOTOS
from tlcv.detectors import FasterRCNN, YOLO11, load_image

torch.set_num_threads(2)


def iou(a, b):
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0])); iy = max(0, min(a[3], b[3]) - max(a[1], b[1])); i = ix * iy
    return i / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - i + 1e-9)

def ioa(det, ign):
    ix = max(0, min(det[2], ign[2]) - max(det[0], ign[0])); iy = max(0, min(det[3], ign[3]) - max(det[1], ign[1]))
    return ix * iy / ((det[2] - det[0]) * (det[3] - det[1]) + 1e-9)


def match(dets, gt):
    out, used = [], set()
    for d in sorted(dets, key=lambda d: -d[4]):
        best, bj = 0, -1
        for j, g in enumerate(gt["lights"]):
            v = iou(d, g["box"])
            if v > best and j not in used: best, bj = v, j
        if best >= 0.5: used.add(bj); out.append((d[4], 1, bj)); continue
        duplicate = any(iou(d, g["box"]) >= 0.5 for g in gt["lights"])
        if not duplicate and any(iou(d, g) >= 0.5 or ioa(d, g) >= 0.5 for g in gt["ignore"]):
            continue
        out.append((d[4], 0, -1))
    return out


def ap_from(scored, n_gt):
    s = sorted(scored, key=lambda x: -x[0]); tp = np.cumsum([x[1] for x in s]); fp = np.cumsum([1 - x[1] for x in s])
    rec = tp / max(n_gt, 1); prec = tp / np.maximum(tp + fp, 1e-9)
    mrec = np.r_[0, rec, 1]; mpre = np.r_[0, prec, 0]
    for i in range(len(mpre) - 2, -1, -1): mpre[i] = max(mpre[i], mpre[i + 1])
    idx = np.where(mrec[1:] != mrec[:-1])[0]
    return float(np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1])), rec, prec


def evaluate(detector, gt, thr=0.30, warm=2):
    imgs = {k: load_image(PHOTOS / k) for k in gt}
    first = next(iter(imgs.values()))
    for _ in range(warm): detector(first)
    scored, ts, per_img, n_gt = [], [], {}, 0
    for k, im in imgs.items():
        t0 = time.perf_counter(); dets = detector(im); ts.append(time.perf_counter() - t0)
        m = match(dets, gt[k]); scored += m; n_gt += len(gt[k]["lights"]); per_img[k] = dets
    ap, rec, prec = ap_from(scored, n_gt)
    at = [x for x in scored if x[0] >= thr]
    tp = sum(x[1] for x in at); fp = len(at) - tp
    small = [(k, j) for k in gt for j, g in enumerate(gt[k]["lights"]) if (g["box"][3] - g["box"][1]) < 60]
    return dict(detector=detector.name, AP50=ap, recall_at_0_3=tp / n_gt, precision_at_0_3=tp / max(tp + fp, 1),
                FP_at_0_3=fp, n_gt=n_gt, latency_ms=1000 * float(np.mean(ts)), FPS=1 / float(np.mean(ts))), per_img


if __name__ == "__main__":
    gt = json.load(open(ROOT / "results" / "scene_gt.json"))["images"]
    rows, all_dets = [], {}
    for D in [FasterRCNN(480, 800), FasterRCNN(960, 1600), YOLO11("n", 1280), YOLO11("s", 1280), YOLO11("m", 1280),
              YOLO11("s", 1920)]:
        if isinstance(D, FasterRCNN): D.name += f" @ {D.m.transform.min_size[0]}"
        r, dets = evaluate(D, gt); rows.append(r); all_dets[D.name] = dets
        print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in r.items()}, flush=True)
    pd.DataFrame(rows).round(4).to_csv(ROOT / "results" / "detector_comparison.csv", index=False)
    json.dump(all_dets, open(ROOT / "results" / "scene_dets.json", "w"))
