# reports/runs

Results of the TRAIN stage: YOLOv8n and YOLOv8s, each trained with two augmentation recipes
(`proposed` and `ultralytics_defaults`), 50 epochs, seed 42, on a Colab T4.
Metrics are on the held-out test split.

**Recommended model:** `yolov8s_proposed/weights/best.pt` (best mAP@0.5:0.95 and F1, real time
on a T4). Fallback without GPU: `yolov8n_proposed/weights/best.pt`.

## Comparison
- `comparacion_recetas.csv`: `proposed` vs `ultralytics_defaults` for both models (metrics and
  per-class mAP@0.5:0.95, with the difference). Latency is not compared between recipes: both
  recipes share the architecture, so their latency gap is noise between two Colab sessions.
- The two recipes differ in more than one thing. `proposed` adds `flipud=0.5` and three
  Albumentations transforms, and passing those through `augmentations=` replaces Ultralytics'
  built-in set (Blur, MedianBlur, ToGray, CLAHE at p=0.01), which `ultralytics_defaults` keeps.
  The `ultralytics_defaults` runs used ultralytics 8.4.173, the `proposed` runs 8.4.172.

## Per-recipe tables
Every file carries the recipe as a suffix (`_proposed`, `_ultralytics_defaults`), as written by
`notebooks/04_training_colab.ipynb` and `scripts/train_eval.py`.
- `tabla_metricas_<recipe>.csv`: mAP, precision, recall and F1 per model (test). Precision and
  recall are the class means Ultralytics reports; F1 is the harmonic mean of those two means.
- `map_por_clase_<recipe>.csv`: mAP@0.5:0.95 per class (test)
- `latencia_t4_<recipe>.csv`: end-to-end latency on a Colab T4 (200 test images, one at a time)
- `latencia_cpu_proposed.csv`: same measurement on a laptop CPU (Intel i5-1135G7), `proposed`
  weights only

## Run folders
`yolov8s_proposed/`, `yolov8n_proposed/`, `yolov8s_ultralytics_defaults/`,
`yolov8n_ultralytics_defaults/`. Each contains:
- `weights/best.pt`: trained weights (use these for inference)
- `args.yaml`: exact training hyperparameters
- `results.csv`, `results.png`: per-epoch training curves
- `confusion_matrix*.png`: computed on the validation split during training
- `test_confusion_matrix*.png`: computed on the test split (see below)
- `Box*_curve.png`: F1, precision, recall and PR curves

## Test confusion matrices
`test_confusion_matrix.png` and `test_confusion_matrix_normalized.png` were generated locally
(ultralytics 8.4.172, CPU) from each `weights/best.pt`:

```python
YOLO("reports/runs/<run>/weights/best.pt").val(
    data="data/processed/data.yaml", split="test", imgsz=640, conf=0.25, plots=True)
```

They use `conf=0.25`, the deployment threshold (also used in the demo), so they count only
detections a deployed system would act on. Ultralytics' validation default is `conf=0.001`, which
is right for mAP but inflates confusions and background false positives; that is the threshold
behind the training-time `confusion_matrix*.png`. Keep the mAP from the tables, not from this
`conf=0.25` pass. The same local evaluation at the default threshold reproduced the Colab test
metrics in `tabla_metricas_*.csv` to four decimals.
