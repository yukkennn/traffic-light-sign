## Results (test split from A1: 77 crops from groups never seen in training)

| Model                            |   seeds |   Params (M) | Accuracy      | Macro-F1      | Yellow recall   | Mean F1 under 11 conditions   | F1 held-out (rain, fog)   |
|:---------------------------------|--------:|-------------:|:--------------|:--------------|:----------------|:------------------------------|:--------------------------|
| ResNet18 — A1 weights (baseline) |       1 |         11.2 | 0.922         | 0.902         | 0.857           | 0.771                         | 0.808                     |
| ResNet18 retrained (A1 recipe)   |       2 |         11.2 | 0.922 ± 0.026 | 0.909 ± 0.031 | 0.857 ± 0.071   | 0.794 ± 0.004                 | 0.810 ± 0.064             |
| ResNet18 + CondSim (A2 final)    |       2 |         11.2 | 0.935 ± 0.013 | 0.922 ± 0.018 | 0.929 ± 0.071   | 0.827 ± 0.024                 | 0.910 ± 0.004             |
| YOLO11s-cls (A1 aug)             |       2 |          5.4 | 0.916 ± 0.019 | 0.895 ± 0.027 | 0.929 ± 0.000   | 0.749 ± 0.025                 | 0.824 ± 0.013             |
| YOLO11s-cls + CondSim            |       2 |          5.4 | 0.864 ± 0.019 | 0.841 ± 0.020 | 0.857 ± 0.000   | 0.761 ± 0.003                 | 0.820 ± 0.044             |

### Ablation — condition simulation (CondSim) on / off

| backbone   | metric                 |   without |   with_condsim |   delta |
|:-----------|:-----------------------|----------:|---------------:|--------:|
| resnet18   | clean macro-F1         |     0.909 |          0.922 |   0.013 |
| resnet18   | mean F1, 11 conditions |     0.794 |          0.827 |   0.033 |
| resnet18   | F1 held-out rain+fog   |     0.81  |          0.91  |   0.101 |
| resnet18   | F1 A1 blur             |     0.714 |          0.791 |   0.076 |
| resnet18   | F1 A1 noise            |     0.599 |          0.624 |   0.025 |
| resnet18   | F1 glare               |     0.689 |          0.783 |   0.095 |
| yolo11s    | clean macro-F1         |     0.895 |          0.841 |  -0.054 |
| yolo11s    | mean F1, 11 conditions |     0.749 |          0.761 |   0.012 |
| yolo11s    | F1 held-out rain+fog   |     0.824 |          0.82  |  -0.004 |
| yolo11s    | F1 A1 blur             |     0.5   |          0.652 |   0.152 |
| yolo11s    | F1 A1 noise            |     0.645 |          0.614 |  -0.03  |
| yolo11s    | F1 glare               |     0.752 |          0.66  |  -0.091 |

### Detectors on 33 own full photos (45 lit lights)

| detector                            |   AP50 |   recall_at_0_3 |   precision_at_0_3 |   latency_ms |
|:------------------------------------|-------:|----------------:|-------------------:|-------------:|
| Faster R-CNN MobileNetV3 (A1) @ 480 |  0.636 |           0.667 |              0.405 |      322.977 |
| Faster R-CNN MobileNetV3 (A1) @ 960 |  0.813 |           0.822 |              0.529 |     1064.59  |
| YOLO11n @ 1280                      |  0.808 |           0.778 |              0.814 |      465.603 |
| YOLO11s @ 1280                      |  0.848 |           0.822 |              0.822 |      939.537 |
| YOLO11m @ 1280                      |  0.885 |           0.844 |              0.808 |     2649.37  |
| YOLO11s @ 1920                      |  0.874 |           0.867 |              0.696 |     2376.53  |

### End-to-end (detector + classifier)

| detector                            | classifier                   |   n_lights |   detected |   state_acc_on_detected |   end_to_end_recall |
|:------------------------------------|:-----------------------------|-----------:|-----------:|------------------------:|--------------------:|
| Faster R-CNN MobileNetV3 (A1) @ 960 | ResNet18 (A1 baseline)       |         45 |         37 |                   1     |               0.822 |
| YOLO11s @ 1280                      | ResNet18 (A1 baseline)       |         45 |         37 |                   0.973 |               0.8   |
| YOLO11s @ 1280                      | YOLO11s-cls (A2 alternative) |         45 |         37 |                   0.757 |               0.622 |
| YOLO11s @ 1280                      | ResNet18 + CondSim (A2)      |         45 |         37 |                   1     |               0.822 |

### Classifier latency (CPU, one crop)

| model            |   params_M |   latency_ms |   crops_per_s |
|:-----------------|-----------:|-------------:|--------------:|
| ResNet18 (A1)    |       11.2 |         57.8 |          17.3 |
| YOLO11s-cls (A2 alternative) |        5.4 |         32.9 |          30.4 |
| YOLO11n-cls      |        1.5 |         14.2 |          70.2 |
