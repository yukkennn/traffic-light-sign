import json, math, random, time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image, ImageEnhance, ImageFilter, ImageDraw, ImageFile
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support, confusion_matrix

ImageFile.LOAD_TRUNCATED_IMAGES = True
ROOT = Path(__file__).resolve().parents[1]
DATA = next((d for d in [ROOT, ROOT.parent, ROOT.parent / "CVAS1"] if (d / "dataset_grouped").exists()), ROOT)
CLASSES = ["red", "yellow", "green"]
MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
SEED = 42
BATCH = 32
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
WEIGHTS = ROOT / "weights"


def set_seed(s=SEED):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)


def load_split(csv=ROOT / "results" / "split_a1.csv"):
    df = pd.read_csv(csv)
    df["path"] = df.path.str.replace("\\", "/", regex=False).map(lambda p: str(DATA / p))
    g = {s: set(df[df.split == s].group) for s in ["train", "val", "test"]}
    assert not (g["train"] & g["val"] or g["train"] & g["test"] or g["val"] & g["test"]), "group leakage"
    return [df[df.split == s].reset_index(drop=True) for s in ["train", "val", "test"]]


def _arr(im): return np.asarray(im.convert("RGB")).astype(np.float32) / 255.0
def _img(a): return Image.fromarray((np.clip(a, 0, 1) * 255).astype(np.uint8))
def _side(im): return max(4, min(im.size))


def c_night(im, s, rng):
    a = _arr(im)
    lum = a.max(2, keepdims=True)
    lamp = np.clip((lum - 0.5) / 0.25, 0, 1) * a
    dark = a ** (1.6 + 1.2 * s) * (0.55 - 0.35 * s)
    out = np.maximum(dark, lamp * (0.9 + 0.1 * rng.random()))
    halo = _arr(_img(lamp).filter(ImageFilter.GaussianBlur(0.08 * _side(im) * (0.5 + s))))
    out = out + 0.8 * s * halo + rng.normal(0, 0.02 + 0.04 * s, a.shape)
    return _img(out)


def c_glare(im, s, rng):
    a = _arr(im)
    lum = a.max(2, keepdims=True)
    lamp = np.clip((lum - 0.45) / 0.3, 0, 1) * a
    halo = _arr(_img(lamp).filter(ImageFilter.GaussianBlur(0.06 * _side(im) * (1 + 2 * s))))
    core = np.clip((lum - 0.55) * (2 + 6 * s), 0, 1)
    out = a * (1 + 0.6 * s) + (1.0 + 1.5 * s) * halo
    out = out * (1 - core) + core * np.array([1.0, 0.97, 0.9])
    return _img(out)


def c_motion(im, s, rng):
    k = max(3, int(round(0.08 * _side(im) * (1 + 3 * s)))) | 1
    ang = rng.uniform(-30, 30) if rng.random() < 0.7 else rng.uniform(60, 120)
    ker = np.zeros((k, k), np.float32); ker[k // 2, :] = 1
    import cv2
    M = cv2.getRotationMatrix2D((k / 2 - .5, k / 2 - .5), ang, 1.0)
    ker = cv2.warpAffine(ker, M, (k, k)); ker /= max(ker.sum(), 1e-6)
    return _img(cv2.filter2D(_arr(im), -1, ker, borderType=cv2.BORDER_REFLECT))


def c_rain(im, s, rng):
    W, H = im.size
    base = ImageEnhance.Contrast(im.convert("RGB")).enhance(1 - 0.35 * s)
    layer = Image.new("L", (W, H), 0); d = ImageDraw.Draw(layer)
    n = int((0.02 + 0.06 * s) * W * H / 4) + 3
    ang = math.radians(rng.uniform(-20, 20)); L = (0.15 + 0.25 * s) * H
    for _ in range(n):
        x, y = rng.uniform(0, W), rng.uniform(-L, H)
        d.line((x, y, x + L * math.sin(ang), y + L * math.cos(ang)), fill=int(rng.uniform(120, 220)), width=1)
    layer = layer.filter(ImageFilter.GaussianBlur(0.6))
    a = _arr(base) * (0.85 - 0.15 * s)
    r = np.asarray(layer, np.float32)[..., None] / 255.0
    out = a * (1 - 0.6 * r) + 0.8 * r
    return _img(out).filter(ImageFilter.GaussianBlur(0.02 * _side(im) * s))


def c_fog(im, s, rng):
    a = _arr(im)
    t = 0.25 + 0.45 * s
    air = np.array([0.78, 0.8, 0.82]) + rng.normal(0, 0.02, 3)
    out = a * (1 - t) + air * t
    return _img(out).filter(ImageFilter.GaussianBlur(0.03 * _side(im) * s))


def c_noise(im, s, rng):
    a = _arr(im) + rng.normal(0, 0.04 + 0.08 * s, (im.size[1], im.size[0], 3))
    import io
    buf = io.BytesIO(); _img(a).save(buf, "JPEG", quality=int(60 - 45 * s)); buf.seek(0)
    return Image.open(buf).convert("RGB")


def c_lowres(im, s, rng):
    W, H = im.size; f = 0.6 - 0.45 * s
    return im.resize((max(4, int(W * f)), max(4, int(H * f))), Image.BILINEAR).resize((W, H), Image.BILINEAR)


CONDITIONS = {"night": c_night, "glare": c_glare, "motion": c_motion, "rain": c_rain,
              "fog": c_fog, "noise+jpeg": c_noise, "low-res": c_lowres}
TRAIN_CONDITIONS = ["night", "glare", "motion", "noise+jpeg", "low-res"]
HELD_OUT = ["rain", "fog"]


class CondSim:
    def __init__(self, p=0.5, conds=TRAIN_CONDITIONS):
        self.p, self.conds = p, conds
    def __call__(self, im):
        if random.random() >= self.p: return im
        rng = np.random.default_rng(random.getrandbits(32))
        return CONDITIONS[random.choice(self.conds)](im.convert("RGB"), rng.uniform(0.2, 1.0), rng)


def a1_blur(im): return im.filter(ImageFilter.GaussianBlur(3))
def a1_dark(im): return ImageEnhance.Brightness(im).enhance(0.35)
def a1_noise(im):
    a = np.asarray(im).astype(np.float32) + np.random.normal(0, 25, (im.size[1], im.size[0], 3))
    return Image.fromarray(a.clip(0, 255).astype(np.uint8))
def a1_lowres(im): return im.resize((24, 24)).resize(im.size)

def fixed(name, s=0.7):
    def f(im, _n=name):
        import zlib
        seed = zlib.crc32(f"{_n}|{im.size}|{int(np.asarray(im).sum())}".encode())
        return CONDITIONS[_n](im.convert("RGB"), s, np.random.default_rng(seed))
    return f

TEST_PERTURB = {"clean": None, "A1 blur": a1_blur, "A1 dark": a1_dark, "A1 noise": a1_noise, "A1 low-res": a1_lowres,
                **{f"sim {c}": fixed(c) for c in CONDITIONS}}


class YoloCls(nn.Module):
    norm = None
    def __init__(self, variant="s", n=len(CLASSES), pretrained=True):
        super().__init__()
        from ultralytics import YOLO
        self.m = YOLO(str(WEIGHTS / f"yolo11{variant}-cls.pt")).model
        head = self.m.model[-1]
        head.linear = nn.Linear(head.linear.in_features, n)
        head.drop.p = 0.2
        for p in self.m.parameters(): p.requires_grad = True
    def head_params(self): return self.m.model[-1].linear.parameters()
    def forward(self, x):
        out = self.m(x)
        return out[1] if isinstance(out, tuple) else out
    @property
    def cam_layer(self): return self.m.model[-2]


class ResNet18(nn.Module):
    norm = (MEAN, STD)
    def __init__(self, n=len(CLASSES), pretrained=True):
        super().__init__()
        self.net = models.resnet18(weights=models.ResNet18_Weights.DEFAULT if pretrained else None)
        self.net.fc = nn.Linear(self.net.fc.in_features, n)
    def head_params(self): return self.net.fc.parameters()
    def forward(self, x): return self.net(x)
    @property
    def cam_layer(self): return self.net.layer4[-1]
    @classmethod
    def from_a1(cls, path=None):
        path = path or next(p for p in [ROOT / "models" / "resnet18.pt", DATA / "models" / "resnet18.pt"] if p.exists())
        m = cls(pretrained=False); m.net.load_state_dict(torch.load(path, map_location="cpu")); return m.eval()


def make_tf(model_cls, size, train, condsim=False):
    t = []
    if train and condsim: t.append(CondSim())
    t.append(transforms.Resize((size, size)))
    if train:
        t += [transforms.RandomAffine(degrees=10, translate=(0.08, 0.08), scale=(0.9, 1.1)),
              transforms.ColorJitter(brightness=0.4, contrast=0.3, saturation=0.2, hue=0.0),
              transforms.RandomApply([transforms.GaussianBlur(5, sigma=(0.1, 2.0))], p=0.3),
              transforms.RandomHorizontalFlip(0.5)]
    t.append(transforms.ToTensor())
    if model_cls.norm: t.append(transforms.Normalize(*model_cls.norm))
    return transforms.Compose(t)


class TLDataset(Dataset):
    def __init__(self, frame, tf, pre=None):
        self.paths, self.y, self.tf, self.pre = frame.path.tolist(), frame.label_id.tolist(), tf, pre
    def __len__(self): return len(self.paths)
    def __getitem__(self, i):
        im = Image.open(self.paths[i]).convert("RGB")
        if self.pre: im = self.pre(im)
        return self.tf(im), self.y[i]


def run_epoch(model, loader, crit, opt=None):
    train = opt is not None; model.train(train)
    tot, ys, ps = 0.0, [], []
    with torch.set_grad_enabled(train):
        for x, y in loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            out = model(x); loss = crit(out, y)
            if train: opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.item() * len(y); ys += y.tolist(); ps += out.argmax(1).tolist()
    return tot / len(ys), accuracy_score(ys, ps), f1_score(ys, ps, average="macro")


def train_model(model, tr, va, epochs, lr, ckpt, class_w, best=-1.0, patience=6, wd=1e-4, log=None, phase=""):
    model.to(DEVICE)
    crit = nn.CrossEntropyLoss(weight=class_w.to(DEVICE), label_smoothing=0.05)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr, weight_decay=wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    hist, bad = [], 0
    for ep in range(1, epochs + 1):
        t0 = time.time()
        tl, ta, tf_ = run_epoch(model, tr, crit, opt)
        vl, va_, vf = run_epoch(model, va, crit)
        sched.step()
        hist.append(dict(phase=phase, epoch=ep, train_loss=tl, val_loss=vl, train_acc=ta, val_acc=va_, train_f1=tf_, val_f1=vf))
        msg = f"{phase} ep {ep:02d} loss {tl:.3f}/{vl:.3f} acc {ta:.3f}/{va_:.3f} valF1 {vf:.3f} ({time.time()-t0:.0f}s)"
        print(msg, flush=True)
        if vf > best: best, bad = vf, 0; torch.save(model.state_dict(), ckpt)
        else:
            bad += 1
            if bad >= patience: break
    return pd.DataFrame(hist), best


@torch.no_grad()
def predict(model, loader):
    model.eval().to(DEVICE); ys, pr = [], []
    for x, y in loader:
        pr.append(model(x.to(DEVICE)).softmax(1).cpu()); ys += y.tolist()
    pr = torch.cat(pr).numpy()
    return np.array(ys), pr.argmax(1), pr


def metrics(y, p):
    P, R, F, _ = precision_recall_fscore_support(y, p, average="macro", zero_division=0)
    pc = precision_recall_fscore_support(y, p, labels=range(3), zero_division=0)
    d = dict(accuracy=accuracy_score(y, p), precision_macro=P, recall_macro=R, f1_macro=F)
    for i, c in enumerate(CLASSES): d[f"f1_{c}"] = pc[2][i]; d[f"recall_{c}"] = pc[1][i]
    return d


def class_weights(train_df):
    cnt = train_df.label_id.value_counts().reindex(range(3)).fillna(1).values
    return torch.tensor(cnt.sum() / (3 * cnt), dtype=torch.float32)
