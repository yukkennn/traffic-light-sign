import platform, time
from pathlib import Path
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

from tlcv.common import CLASSES, DATA, ROOT, WEIGHTS, ResNet18, YoloCls, make_tf
from tlcv.detectors import FasterRCNN, YOLO11

COLORS = {"red": (255, 59, 59), "yellow": (255, 200, 0), "green": (40, 210, 90)}
UNSURE = (170, 170, 170)
CLASSIFIERS = {"ResNet18 + CondSim (A2)": ("resnet18", "resnet18_condsim_s42.pt"),
               "ResNet18 (A1 baseline)": ("a1_resnet18", "resnet18.pt"),
               "YOLO11s-cls (A2 alternative)": ("yolo11s", "yolo11s_s42.pt")}
DETECTORS = {"YOLO11s @1280 (A2)": lambda: YOLO11("s", 1280, 0.10),
             "Faster R-CNN MobileNetV3 @960 (A1 baseline)": lambda: FasterRCNN(960, 1600, 0.10)}


def pad_box(b, W, H, margin=0.10, min_side=8):
    x0, y0, x1, y1 = b[:4]; w, h = max(x1 - x0, 1), max(y1 - y0, 1)
    mx, my = max(w * margin, (min_side - w) / 2, 0), max(h * margin, (min_side - h) / 2, 0)
    return (max(0, int(x0 - mx)), max(0, int(y0 - my)), min(W, int(x1 + mx) + 1), min(H, int(y1 + my) + 1))


def lamp_point(crop):
    hsv = np.asarray(crop.convert("HSV")).astype(np.float32)
    score = hsv[..., 2] * (0.4 + hsv[..., 1] / 255.0)
    if score.max() < 120: return None
    y, x = np.unravel_index(np.argmax(score), score.shape)
    return int(x), int(y)


def lamp_evidence(crop, s_min=90, v_min=140, min_frac=0.01, size=48):
    hsv = np.asarray(crop.convert("RGB").resize((size, size)).convert("HSV")).astype(np.int16)
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    lit = (s >= s_min) & (v >= v_min); n = int(lit.sum())
    if n < min_frac * lit.size: return dict(lit=False, vote=None)
    hl = h[lit]
    cnt = {"red": int(((hl <= 18) | (hl >= 236)).sum()), "yellow": int(((hl >= 19) & (hl <= 50)).sum()),
           "green": int(((hl >= 51) & (hl <= 140)).sum())}
    top = max(cnt, key=cnt.get)
    return dict(lit=True, vote=top if cnt[top] >= 0.6 * n else None)


class Engine:
    def __init__(self):
        self._cls, self._det, self._sam = {}, {}, None
        torch.set_num_threads(max(1, torch.get_num_threads()))

    def classifier(self, key):
        if key not in self._cls:
            kind, f = CLASSIFIERS[key]
            f = ROOT / "models" / f if (ROOT / "models" / f).exists() else DATA / "models" / f
            if kind == "a1_resnet18": m, M = ResNet18.from_a1(f), ResNet18
            elif kind == "resnet18":
                m, M = ResNet18(pretrained=False), ResNet18
                m.load_state_dict(torch.load(f, map_location="cpu"))
            else:
                m, M = YoloCls(kind[-1]), YoloCls
                m.load_state_dict(torch.load(f, map_location="cpu"))
            self._cls[key] = (m.eval(), make_tf(M, 224, False))
        return self._cls[key]

    def detector(self, key):
        if key not in self._det: self._det[key] = DETECTORS[key]()
        return self._det[key]

    def sam(self):
        if self._sam is None:
            from ultralytics import SAM
            self._sam = SAM(str(WEIGHTS / "sam2.1_t.pt"))
        return self._sam

    @torch.no_grad()
    def classify(self, key, crops):
        if not crops: return []
        m, tf = self.classifier(key)
        p = m(torch.stack([tf(c) for c in crops])).softmax(1).numpy()
        return [dict(zip(CLASSES, map(float, r))) for r in p]

    def gradcam(self, key, crop, target=None):
        from pytorch_grad_cam import GradCAM
        from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
        m, tf = self.classifier(key)
        x = tf(crop).unsqueeze(0)
        with GradCAM(model=m, target_layers=[m.cam_layer]) as cam:
            g = cam(input_tensor=x, targets=[ClassifierOutputTarget(target)] if target is not None else None)[0]
        return g

    def sam_lamp_mask(self, img, box):
        W, H = img.size; pb = pad_box(box, W, H, margin=0.25)
        crop = img.crop(pb); cw, ch = crop.size
        scale = 256 / max(cw, ch) if max(cw, ch) < 256 else 1.0
        big = crop.resize((max(1, int(cw * scale)), max(1, int(ch * scale))), Image.BICUBIC)
        bx = [(box[0] - pb[0]) * scale, (box[1] - pb[1]) * scale, (box[2] - pb[0]) * scale, (box[3] - pb[1]) * scale]
        sam = self.sam()
        house = sam.predict(big, bboxes=[bx], verbose=False, device="cpu")[0].masks
        house = house.data[0].numpy().astype(bool) if house is not None else np.zeros(big.size[::-1], bool)
        lamp = np.zeros_like(house)
        pt = lamp_point(big.crop((int(bx[0]), int(bx[1]), int(bx[2]), int(bx[3]))))
        if pt is not None:
            pt = [pt[0] + bx[0], pt[1] + bx[1]]
            r = sam.predict(big, points=[pt], labels=[1], verbose=False, device="cpu")[0].masks
            if r is not None:
                lamp = r.data[0].numpy().astype(bool)
                if lamp.sum() > 0.6 * max(house.sum(), 1): lamp &= house
        return pb, big, house, lamp, pt

    def analyse(self, img, det_key, cls_key, det_thr=0.30, min_conf=0.5):
        W, H = img.size; t = {}
        t0 = time.perf_counter(); raw = self.detector(det_key)(img); t["detect_ms"] = 1000 * (time.perf_counter() - t0)
        dets = [d for d in raw if d[4] >= det_thr]
        if len(dets) > 1:
            from torchvision.ops import nms
            b = torch.tensor([d[:4] for d in dets], dtype=torch.float32); sc = torch.tensor([d[4] for d in dets])
            dets = [dets[i] for i in nms(b, sc, 0.4).tolist()]
        crops = [img.crop(pad_box(d, W, H)) for d in dets]
        t0 = time.perf_counter(); probs = self.classify(cls_key, crops); t["classify_ms"] = 1000 * (time.perf_counter() - t0)
        out = []
        for d, c, p in zip(dets, crops, probs):
            lab = max(p, key=p.get); ev = lamp_evidence(c)
            status = "ok"
            if not ev["lit"]: status = "no lit lamp"
            elif p[lab] < min_conf: status = "low confidence"
            elif ev["vote"] and ev["vote"] != lab: status = f"colour check says {ev['vote']}"
            out.append(dict(box=[float(v) for v in d[:4]], det_score=float(d[4]), label=lab, conf=p[lab], probs=p,
                            status=status, crop=c))
        t["n_crops"] = len(crops)
        return out, t


def font(size):
    for n in ("arial.ttf", "DejaVuSans-Bold.ttf", "DejaVuSans.ttf"):
        try: return ImageFont.truetype(n, size)
        except OSError: pass
    return ImageFont.load_default()


def draw(img, dets, show_unlit=False, names=None):
    im = img.copy(); d = ImageDraw.Draw(im); W, H = im.size
    f = font(max(14, int(min(W, H) / 45))); lw = max(2, W // 400)
    for k, det in enumerate(dets, 1):
        if det["status"] == "no lit lamp" and not show_unlit: continue
        sure = det["status"] == "ok"; col = COLORS[det["label"]] if sure else UNSURE
        text = f"{k}: {(names or {}).get(det['label'], det['label'])} {det['conf']:.0%}" + ("" if sure else " ?")
        x0, y0, x1, y1 = det["box"]; d.rectangle((x0, y0, x1, y1), outline=col, width=lw)
        tb = d.textbbox((0, 0), text, font=f); th, tw = tb[3] - tb[1] + 8, tb[2] - tb[0] + 10
        ty = y0 - th if y0 - th >= 0 else y1
        d.rectangle((x0, ty, x0 + tw, ty + th), fill=col); d.text((x0 + 5, ty + 2), text, fill=(0, 0, 0), font=f)
    return im


def heat_overlay(crop, cam, size=224, alpha=0.45):
    import matplotlib; cmap = matplotlib.colormaps["jet"]
    base = np.asarray(crop.convert("RGB").resize((size, size)), np.float32) / 255
    h = cmap(cam)[..., :3]
    return Image.fromarray((255 * ((1 - alpha) * base + alpha * h)).astype(np.uint8))


def mask_overlay(big, house, lamp, pt=None):
    a = np.asarray(big.convert("RGB"), np.float32) / 255
    out = a.copy()
    out[house] = 0.6 * a[house] + 0.4 * np.array([0.2, 0.6, 1.0])
    out[lamp] = 0.35 * a[lamp] + 0.65 * np.array([1.0, 0.0, 1.0])
    im = Image.fromarray((255 * out).astype(np.uint8))
    if pt is not None:
        d = ImageDraw.Draw(im); r = max(3, big.size[0] // 60)
        d.ellipse((pt[0] - r, pt[1] - r, pt[0] + r, pt[1] + r), outline="white", width=2)
    return im


def watermark(im, text="SYNTHETIC - simulated condition"):
    im = im.convert("RGB").copy(); d = ImageDraw.Draw(im); W, H = im.size
    f = font(max(10, W // 18)); tb = d.textbbox((0, 0), text, font=f)
    d.rectangle((0, H - (tb[3] - tb[1]) - 8, W, H), fill=(0, 0, 0)); d.text((4, H - (tb[3] - tb[1]) - 6), text, fill=(255, 220, 0), font=f)
    return im


def environment():
    import os
    cpu = platform.processor() or platform.machine()
    return dict(device="CUDA " + torch.cuda.get_device_name(0) if torch.cuda.is_available() else f"CPU ({cpu}, {os.cpu_count()} threads)",
                os=f"{platform.system()} {platform.release()}", python=platform.python_version(), torch=torch.__version__)
