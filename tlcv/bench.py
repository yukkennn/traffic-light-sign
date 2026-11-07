import time
import numpy as np, pandas as pd, torch
from tlcv.common import ROOT, ResNet18, YoloCls

torch.set_num_threads(max(1, torch.get_num_threads()))
rows = []
for name, m in [("ResNet18 (A1)", ResNet18.from_a1()), ("YOLO11s-cls (A2 alternative)", YoloCls("s")), ("YOLO11n-cls", YoloCls("n"))]:
    m.eval(); x = torch.randn(1, 3, 224, 224)
    with torch.no_grad():
        for _ in range(10): m(x)
        ts = []
        for _ in range(100):
            t0 = time.perf_counter(); m(x); ts.append(time.perf_counter() - t0)
    rows.append(dict(model=name, params_M=sum(p.numel() for p in m.parameters()) / 1e6,
                     latency_ms=1000 * np.median(ts), crops_per_s=1 / np.median(ts)))
    print(rows[-1], flush=True)
pd.DataFrame(rows).round(2).to_csv(ROOT / "results" / "latency.csv", index=False)
