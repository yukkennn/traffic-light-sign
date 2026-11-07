import glob, json
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tlcv.common import ROOT, CLASSES, HELD_OUT

R = ROOT / "results"; F = R / "figures"; F.mkdir(exist_ok=True)
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False, "figure.dpi": 150})
runs = [json.load(open(f)) for f in sorted(glob.glob(str(R / "runs" / "*.json")))]
df = pd.DataFrame([dict(config=r["name"].rsplit("_s", 1)[0], seed=r["seed"], params=r["params"],
                        **{k: r["test"][k] for k in ["accuracy", "precision_macro", "recall_macro", "f1_macro",
                                                     "recall_yellow", "f1_yellow", "f1_red", "f1_green"]},
                        **{f"rob|{k}": v for k, v in r["robustness_f1"].items()}) for r in runs])
NAMES = {"a1_resnet18": "ResNet18 — A1 weights (baseline)", "resnet18": "ResNet18 retrained (A1 recipe)",
         "resnet18_condsim": "ResNet18 + CondSim (A2 final)", "yolo11s": "YOLO11s-cls (A1 aug)",
         "yolo11s_condsim": "YOLO11s-cls + CondSim", "yolo11n_condsim": "YOLO11n-cls + CondSim"}
ORDER = [k for k in NAMES if k in set(df.config)]
COL = {"a1_resnet18": "#9a9a9a", "resnet18": "#c9c9c9", "resnet18_condsim": "#DD8452",
       "yolo11s": "#4C72B0", "yolo11s_condsim": "#9DB6DD", "yolo11n_condsim": "#8172B2"}
rob_cols = [c for c in df.columns if c.startswith("rob|") and c != "rob|clean"]
df["rob_mean"] = df[rob_cols].mean(axis=1)
df["rob_heldout"] = df[[f"rob|sim {c}" for c in HELD_OUT]].mean(axis=1)


def ms(g, c): return f"{g[c].mean():.3f} ± {g[c].std(ddof=0):.3f}" if len(g) > 1 else f"{g[c].mean():.3f}"


t1 = []
for k in ORDER:
    g = df[df.config == k]
    t1.append({"Model": NAMES[k], "seeds": len(g), "Params (M)": f"{g.params.iloc[0]/1e6:.1f}", "Accuracy": ms(g, "accuracy"),
               "Macro-F1": ms(g, "f1_macro"), "Yellow recall": ms(g, "recall_yellow"),
               "Mean F1 under 11 conditions": ms(g, "rob_mean"), "F1 held-out (rain, fog)": ms(g, "rob_heldout")})
t1 = pd.DataFrame(t1); t1.to_csv(R / "table_classifiers.csv", index=False)

rob = df.groupby("config")[["rob|clean"] + rob_cols].mean().reindex(ORDER)
rob.columns = [c.split("|")[1] for c in rob.columns]; rob.index = [NAMES[k] for k in rob.index]
rob.round(3).to_csv(R / "table_robustness.csv")

abl = []
for bb in ["resnet18", "yolo11s"]:
    a, b = df[df.config == bb], df[df.config == f"{bb}_condsim"]
    if len(a) and len(b):
        for col, lab in [("f1_macro", "clean macro-F1"), ("rob_mean", "mean F1, 11 conditions"), ("rob_heldout", "F1 held-out rain+fog"),
                         ("rob|A1 blur", "F1 A1 blur"), ("rob|A1 noise", "F1 A1 noise"), ("rob|sim glare", "F1 glare")]:
            abl.append(dict(backbone=bb, metric=lab, without=a[col].mean(), with_condsim=b[col].mean(), delta=b[col].mean() - a[col].mean()))
abl = pd.DataFrame(abl); abl.round(3).to_csv(R / "table_ablation.csv", index=False)

fig, ax = plt.subplots(1, 3, figsize=(11, 3))
for k in ["resnet18", "resnet18_condsim", "yolo11s", "yolo11s_condsim"]:
    col = COL[k]
    p = R / "runs" / f"{k}_s42_history.csv"
    if not p.exists(): continue
    h = pd.read_csv(p); x = np.arange(1, len(h) + 1)
    ax[0].plot(x, h.train_loss, color=col, label=f"{NAMES[k]} train"); ax[0].plot(x, h.val_loss, "--", color=col)
    ax[1].plot(x, h.train_acc, color=col); ax[1].plot(x, h.val_acc, "--", color=col)
    ax[2].plot(x, h.val_f1, color=col, label=NAMES[k])
    n_head = (h.phase == "head").sum()
for a, t in zip(ax, ["Loss (solid train, dashed val)", "Accuracy (solid train, dashed val)", "Validation macro-F1"]):
    a.set_title(t); a.set_xlabel("epoch"); a.axvline(n_head + .5, color="k", lw=.6, ls=":")
ax[2].legend(fontsize=7, loc="lower right"); plt.tight_layout(); plt.savefig(F / "training_curves.png"); plt.close()

cols = list(rob.columns); selk = [k for k in ORDER if k not in ("resnet18", "yolo11n_condsim")]; sel = [NAMES[k] for k in selk]
fig, ax = plt.subplots(figsize=(11, 3.8)); w = 0.8 / len(sel)
for i, (k, n) in enumerate(zip(selk, sel)):
    ax.bar(np.arange(len(cols)) + i * w - 0.4 + w / 2, rob.loc[n], w, label=n, color=COL[k])
ax.set_xticks(range(len(cols))); ax.set_xticklabels([c + (" *" if c.replace("sim ", "") in HELD_OUT else "") for c in cols], rotation=25, ha="right")
ax.set_ylim(0.4, 1.0); ax.set_ylabel("macro-F1 (test)")
ax.legend(fontsize=7, ncol=4, loc="lower center", bbox_to_anchor=(0.5, 1.0), frameon=False)
ax.set_title("Robustness on the test crops (mean over seeds). * = condition never used in training", pad=22)
plt.tight_layout(); plt.savefig(F / "robustness.png"); plt.close()

fig, axs = plt.subplots(1, 2, figsize=(7, 3))
for a, k in zip(axs, ["a1_resnet18_s42", "resnet18_condsim_s42"]):
    p = R / "runs" / f"{k}.json"
    if not p.exists(): continue
    cm = np.array(json.load(open(p))["cm"]); a.imshow(cm, cmap="Blues")
    for i in range(3):
        for j in range(3): a.text(j, i, cm[i, j], ha="center", va="center", color="white" if cm[i, j] > cm.max() / 2 else "black")
    a.set_xticks(range(3)); a.set_yticks(range(3)); a.set_xticklabels(CLASSES); a.set_yticklabels(CLASSES)
    a.set_xlabel("predicted"); a.set_ylabel("true"); a.set_title(NAMES[k.rsplit("_s", 1)[0]], fontsize=8)
plt.tight_layout(); plt.savefig(F / "confusion.png"); plt.close()

det = pd.read_csv(R / "detector_comparison.csv") if (R / "detector_comparison.csv").exists() else None
if det is not None:
    fig, ax = plt.subplots(figsize=(5.5, 3.2))
    for _, r in det.iterrows():
        c = "#9a9a9a" if "Faster" in r.detector else "#DD8452"
        ax.scatter(r.latency_ms, r.AP50, s=60, color=c, zorder=3)
        dy = -10 if "1920" in r.detector else 4
        ax.annotate(r.detector.replace("MobileNetV3 (A1) ", "(A1) "), (r.latency_ms, r.AP50), fontsize=7, xytext=(-60 if "1920" in r.detector else 5, dy), textcoords="offset points")
    ax.set_xscale("log"); ax.set_xlabel("CPU latency per image, ms (log)"); ax.set_ylabel("AP@0.5 (lit traffic lights)")
    ax.set_title("Detectors on 33 own full-resolution photos"); ax.grid(alpha=.3)
    plt.tight_layout(); plt.savefig(F / "detectors.png"); plt.close()

md = ["## Results (test split from A1: 77 crops from groups never seen in training)", "", t1.to_markdown(index=False), ""]
if len(abl): md += ["### Ablation — condition simulation (CondSim) on / off", "", abl.round(3).to_markdown(index=False), ""]
if det is not None:
    md += ["### Detectors on 33 own full photos (45 lit lights)", "",
           det[["detector", "AP50", "recall_at_0_3", "precision_at_0_3", "latency_ms"]].round(3).to_markdown(index=False), ""]
if (R / "end_to_end.csv").exists():
    md += ["### End-to-end (detector + classifier)", "", pd.read_csv(R / "end_to_end.csv").round(3).to_markdown(index=False), ""]
if (R / "latency.csv").exists():
    md += ["### Classifier latency (CPU, one crop)", "", pd.read_csv(R / "latency.csv").round(1).to_markdown(index=False), ""]
(R / "summary.md").write_text("\n".join(md), encoding="utf-8")
print("\n".join(md)); print(rob.round(3).to_string())
