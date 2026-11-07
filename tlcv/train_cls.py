import argparse, json, time
import numpy as np, pandas as pd, torch
from torch.utils.data import DataLoader
from tlcv.common import *

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="yolo11s", choices=["yolo11s", "yolo11n", "resnet18", "a1_resnet18"])
ap.add_argument("--condsim", action="store_true")
ap.add_argument("--seed", type=int, default=42)
ap.add_argument("--size", type=int, default=224)
ap.add_argument("--epochs_head", type=int, default=5)
ap.add_argument("--epochs_ft", type=int, default=20)
a = ap.parse_args()
torch.set_num_threads(2)

name = f"{a.model}{'_condsim' if a.condsim else ''}_s{a.seed}"
out = ROOT / "results" / "runs"; out.mkdir(parents=True, exist_ok=True)
ckpt = ROOT / "models" / f"{name}.pt"
train_df, val_df, test_df = load_split()
set_seed(a.seed)

if a.model == "a1_resnet18":
    model, Mcls, hist = ResNet18.from_a1(), ResNet18, None
else:
    Mcls = {"yolo11s": YoloCls, "yolo11n": YoloCls, "resnet18": ResNet18}[a.model]
    model = YoloCls(a.model[-1]) if Mcls is YoloCls else ResNet18()
    mk = lambda d, train: DataLoader(TLDataset(d, make_tf(Mcls, a.size, train, a.condsim and train)),
                                     batch_size=BATCH, shuffle=train, num_workers=2)
    tr, va = mk(train_df, True), mk(val_df, False)
    cw = class_weights(train_df)
    t0 = time.time()
    for p in model.parameters(): p.requires_grad = False
    for p in model.head_params(): p.requires_grad = True
    h1, best = train_model(model, tr, va, a.epochs_head, 1e-3, ckpt, cw, phase="head")
    for p in model.parameters(): p.requires_grad = True
    h2, best = train_model(model, tr, va, a.epochs_ft, 1e-4, ckpt, cw, best=best, phase="ft")
    hist = pd.concat([h1, h2], ignore_index=True); hist.to_csv(out / f"{name}_history.csv", index=False)
    model.load_state_dict(torch.load(ckpt, map_location="cpu")); train_time = time.time() - t0

res = dict(name=name, model=a.model, condsim=a.condsim, seed=a.seed,
           params=sum(p.numel() for p in model.parameters()))
if hist is not None: res["train_time_s"] = train_time; res["best_val_f1"] = float(hist.val_f1.max())

tf = make_tf(Mcls, a.size, False)
rob, cms = {}, {}
for pn, fn in TEST_PERTURB.items():
    np.random.seed(0); torch.manual_seed(0)
    y, p, pr = predict(model, DataLoader(TLDataset(test_df, tf, pre=fn), batch_size=BATCH, num_workers=0))
    m = metrics(y, p); rob[pn] = m["f1_macro"]
    if pn == "clean":
        res["test"] = m; res["cm"] = confusion_matrix(y, p, labels=range(3)).tolist()
        res["test_pred"] = p.tolist(); res["test_conf"] = pr.max(1).round(4).tolist()
res["robustness_f1"] = rob

model.eval(); x = torch.randn(1, 3, a.size, a.size)
with torch.no_grad():
    for _ in range(5): model(x)
    ts = []
    for _ in range(50):
        t0 = time.perf_counter(); model(x); ts.append(time.perf_counter() - t0)
res["latency_ms_cpu"] = 1000 * float(np.median(ts))
json.dump(res, open(out / f"{name}.json", "w"), indent=1)
print(json.dumps({k: res[k] for k in ["name", "params", "latency_ms_cpu"]}), json.dumps(res["test"]), json.dumps(rob))
