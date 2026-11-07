# Traffic Light State Recognition — Assignment 2 (SOTA upgrade + web demo)

A2 continues the A1 project (red / yellow / green traffic-light state from images; 421 crops, group-wise split)
and adds **state-of-the-art components**, a **web app**, and **explainability**:

| Stage | A1 baseline | A2 (SOTA) |
|---|---|---|
| Detector | Faster R-CNN MobileNetV3 (torchvision, 2019) | **YOLO11** (Ultralytics, 2024) |
| State classifier | ResNet18 transfer learning | **ResNet18 + CondSim** (final); YOLO11s-cls (C3k2 + C2PSA) tested as an alternative |
| Robustness | mild blur augmentation | **CondSim**: night / glare / motion / noise+JPEG / low-res simulation (rain & fog held out) |
| Explanation | — | **Grad-CAM** heat-maps + **SAM 2.1** lamp / housing masks (promptable segmentation) |
| App | tkinter desktop | **Gradio web app**: upload or camera → prediction → explanation, latency + environment |

All numbers in `results/` were produced by the code in this repository (CPU only).
`results/summary.md` holds the main tables.

## Quick start

The repository is self-contained: the A1 dataset (`dataset_grouped/`, 421 crops) and the A1 model (`models/resnet18.pt`)
are included.

```bash
git clone https://github.com/yukkennn/traffic-light-sign
cd traffic-light-sign

# create an environment (Windows; on Linux/macOS: source .venv/bin/activate)
python -m venv .venv
.venv\Scripts\activate

pip install -r requirements.txt
python get_weights.py      # YOLO11 + SAM 2.1 weights, ~165 MB, once
python app.py              # opens http://127.0.0.1:7860
```
On Windows you can also double-click `run_app_a2.bat`; on first start it installs missing libraries and downloads the weights. Switch the interface language with the EN / RU button in the top-right corner. Use `python app.py --share` to get a temporary public link (for a phone).
Included models: `models/resnet18_condsim_s42.pt` (A2, final), `models/yolo11s_s42.pt` (A2 alternative) and `models/resnet18.pt` (A1 baseline). All three are stored in half precision to keep the files small; predictions are unchanged.

## Web app
* **Recognise**: add a photo (upload, webcam / phone camera, or an example) and press *Find traffic lights*. Each light gets a card
  with a signal icon, its name, a confidence bar and a plain-language check. *Why did the model decide this?* shows the Grad-CAM
  heat-map and, if enabled, the SAM 2 mask of the housing (blue) and the lit lamp (magenta). *Details* lists the probabilities, the speed of each step,
  the device, OS and browser. The detector, classifier (A2 / A1 / YOLO11s-cls), thresholds and explanations are in *Settings*.
* **Bad-weather simulator**: apply a simulated condition to a traffic-light close-up and compare all three classifiers side by side.
  Every simulated image is watermarked *SYNTHETIC*.
* **Results and ethics**: key numbers, disclosure and limitations, all result tables.

## Reproducing the experiments
```bash
python -m tlcv.train_cls --model a1_resnet18                 # evaluate the A1 weights on the A1 split
python -m tlcv.train_cls --model yolo11s  --seed 42          # YOLO11s-cls, A1 augmentation
python -m tlcv.train_cls --model yolo11s  --condsim --seed 42
python -m tlcv.train_cls --model resnet18 [--condsim] --seed 42
bash run_all.sh                                              # all classifier runs used in the report (seeds 42 and 7)
python -m tlcv.det_eval      # detectors on 33 own full photos -> results/detector_comparison.csv
python -m tlcv.e2e_eval      # detector x classifier end-to-end
python -m tlcv.xai_eval      # Grad-CAM pointing game against SAM 2 lamp masks
python -m tlcv.det_latency   # clean detector latency (idle CPU)
python -m tlcv.bench         # clean classifier latency
python -m tlcv.summarize     # tables + figures -> results/
```
* Split: `results/split_a1.csv` is the exact A1 split (258 train / 86 val / 77 test, 10 groups). No group appears in two subsets.
* Training recipe (same as A1): class-weighted CE with label smoothing 0.05, AdamW (wd 1e-4), cosine LR, head-only 5 epochs at 1e-3,
  then full fine-tune ≤20 epochs at 1e-4, early stopping on validation macro-F1.
* `results/scene_gt.json`: boxes and states for the 45 lit traffic lights in the author's 33 street photos. Unlit, back-facing and pedestrian lights are
  marked *ignore*. These boxes were checked by eye from the union of detector proposals, so a light that every detector missed is not in the set.
* **The 33 street photos are not published** (privacy: cars, plates and people). Their results are saved in `results/`
  (`detector_comparison.csv`, `end_to_end.csv`, `scene_dets.json`). To rerun `det_eval`, `det_latency` and `e2e_eval`, put your own photos
  in `images/full/` (the ground truth then has to be redone). Everything else runs from this repository alone.

## Repository layout
```
app.py                 Gradio web app
report/                A2 report (PDF + DOCX) and the script that builds it
tlcv/common.py         data split, CondSim condition simulation, models, training loop, metrics
tlcv/detectors.py      Faster R-CNN (A1) and YOLO11 wrappers
tlcv/pipeline.py       detect -> crop -> classify -> Grad-CAM / SAM 2 explanation
tlcv/train_cls.py      one classifier experiment (writes results/runs/*.json)
tlcv/det_eval.py, det_latency.py, e2e_eval.py, xai_eval.py, bench.py, summarize.py
dataset_grouped/       A1 dataset: 421 crops (red / yellow / green), 10 groups per class
models/                A1 baseline + A2 classifier weights
results/               metrics (CSV/JSON), figures/, summary.md
weights/               YOLO11 + SAM 2.1 weights, downloaded by get_weights.py
```
The A1 code (`traffic_light_app.py`, notebook, crop tool) stays unchanged in the parent project folder.

## Ethics & disclosure
This is a student prototype, **not** a driving aid. The simulated-condition images are procedural (no generative model) and are
watermarked in the app. Uploaded images are not stored. The data covers daylight scenes from a few cities only.
Credits: S2TLD dataset; Ultralytics YOLO11 & SAM 2.1 (Meta); pytorch-grad-cam; torchvision.
