# reports/runs

Results of the TRAIN stage: YOLOv8n and YOLOv8s, each trained with two augmentation recipes
(`proposed` and `ultralytics_defaults`), 50 epochs, seed 42, on a Colab T4.
Metrics are on the held-out test split.

**Recommended model:** `yolov8s_proposed/weights/best.pt` (best mAP@0.5:0.95 and F1, real time
on a T4). Fallback without GPU: `yolov8n_proposed/weights/best.pt`.

## Comparison
- `comparacion_recetas.csv`: `proposed` vs `ultralytics_defaults` for both models (metrics,
  latency and per-class mAP@0.5:0.95, with the difference).

## Per-recipe tables
Files without a suffix belong to the `proposed` run. Re-running the notebook names the files of
every recipe with a suffix (`_proposed`, `_ultralytics_defaults`).
- `tabla_metricas*.csv`: mAP, precision, recall and F1 per model (test)
- `map_por_clase*.csv`: mAP@0.5:0.95 per class (test)
- `latencia_t4*.csv`: end-to-end latency on a Colab T4 (200 test images, one at a time)
- `latencia_cpu.csv`: same measurement on a laptop CPU (Intel i5-1135G7), `proposed` weights only

## Run folders
`yolov8s_proposed/`, `yolov8n_proposed/`, `yolov8s_ultralytics_defaults/`,
`yolov8n_ultralytics_defaults/`. Each contains:
- `weights/best.pt`: trained weights (use these for inference)
- `args.yaml`: exact training hyperparameters
- `results.csv`, `results.png`: per-epoch training curves
- `confusion_matrix*.png`: computed on the validation split, not on test
- `Box*_curve.png`: F1, precision, recall and PR curves
