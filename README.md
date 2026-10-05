# VPC2

Computer vision pipeline for recyclable waste detection on a conveyor belt
(CEIA, Visión por Computadora II). The first stage — data processing — audits
the Roboflow YOLO export, cleans labels, and writes a held-out train/val/test
split. Shared code lives in `src/vpc2/`; notebooks and scripts call it.

## Requirements

- Python 3.12+
- [uv](https://docs.astral.sh/uv/)

## Setup

```bash
uv sync --group dev
```

This creates `.venv/` and installs the project plus pytest/ruff.

## Dataset

Download [GARBAGE CLASSIFICATION 3](https://universe.roboflow.com/material-identification/garbage-classification-3)
**v2**, YOLOv8 format, and extract it under `data/raw/`. Git ignores that
directory. Expected layout:

```
data/raw/<export>/
  data.yaml
  train|valid|test/
    images/
    labels/
```

v2 has 10,464 images across six classes: biodegradable, cardboard, glass,
metal, paper, plastic.

## Data processing

Raw files are never modified. The pipeline writes audit manifests to
`data/interim/` and a training-ready YOLO layout to `data/processed/`.

```bash
uv run python scripts/process_data.py
```

Same flow in `notebooks/01_data_processing.ipynb` (`uv run jupyter lab`).

Outputs:

- `data/interim/audit/report.json` — counts, class histogram, warnings
- `data/processed/data.yaml` — Ultralytics config (train / val / **held-out test**)
- `data/processed/images/{train,val,test}/` and matching `labels/`

Do not augment the test split.

## Augmentation

Augmentation is applied **online, during training**, from one shared policy file, so every
teammate trains with identical settings and there are no augmented copies to keep in sync.

- `configs/augmentation.yaml` — the policy, one arm per ablation experiment
  (`no_augmentation`, `ultralytics_defaults`, `proposed`, `proposed_low_saturation`).
- `notebooks/03_augmentation_analysis.ipynb` — the measurements behind each choice, a
  preview of the policy, and the export below.
- `data/augmented/` — a small **preview** sample (train images only) written by that
  notebook. For looking at the policy, never for training.

```python
from vpc2.data.augmentation import load_policy

policy = load_policy("proposed")
model.train(data="data/processed/data.yaml", **policy.train_kwargs())
```

Name the baseline arm "Ultralytics defaults" (Ultralytics augments by default), not "no
augmentation".

## Training & Evaluation

Fine-tunes pretrained YOLO variants (YOLOv8n and YOLOv8s) with the shared augmentation policy
(`configs/augmentation.yaml`), evaluates them on the held-out **test** split and measures latency.
Requires `data/processed/` (see Data processing) and a GPU for training (a Colab T4 works; on CPU
it would take days).

- **Colab:** `notebooks/04_training_colab.ipynb`. Edit the config cell (Drive folder with the two
  zips, `POLICY`, `MODELS`) and run the cells in order.
- **Local GPU:** `uv run python scripts/train_eval.py` (options: `--policy`, `--weights`,
  `--epochs`, `--batch`). `--skip-train` re-evaluates saved weights without training. The
  evaluation mode reproduced the Colab metrics; its training mode mirrors the notebook but has
  not been run end to end.
- The reported runs were made on a Colab T4 with the same parameters as the notebook (50 epochs,
  batch 16, seed 42). Their numbers, curves and hyperparameters are in `reports/runs/`.

Outputs go to `reports/runs/` (see its README).

### Augmentation recipes trained

| Recipe | What it is |
|---|---|
| `ultralytics_defaults` | Baseline: Ultralytics' own default augmentation (mosaic, HSV jitter, horizontal flip, translate, scale). |
| `proposed` | The defaults plus vertical flip, darkening (brightness down to -40%), motion blur and Gaussian blur, chosen from gaps measured in the data (see notebook 03). |

`no_augmentation` and `proposed_low_saturation` are defined in `configs/augmentation.yaml` but have not been trained.

### Results (test split, 50 epochs, seed 42)

| Model | Recipe | mAP@0.5 | mAP@0.5:0.95 | F1 | Latency (Colab T4) |
|---|---|---|---|---|---|
| YOLOv8s | proposed | 0.650 | 0.467 | 0.650 | 18.1 ms (55 FPS) |
| YOLOv8s | ultralytics_defaults | 0.645 | 0.455 | 0.646 | 19.3 ms (52 FPS) |
| YOLOv8n | proposed | 0.629 | 0.443 | 0.625 | 10.2 ms (98 FPS) |
| YOLOv8n | ultralytics_defaults | 0.622 | 0.435 | 0.617 | 10.4 ms (96 FPS) |

Latency is end to end per image (one at a time, 200 test images). On a laptop CPU (Intel
i5-1135G7, measured on the `proposed` weights): YOLOv8n 68.9 ms (14.5 FPS), YOLOv8s 177.3 ms
(5.6 FPS). Per-class values and all differences: `reports/runs/comparacion_recetas.csv`.

- `proposed` beats the baseline by +1.2 (YOLOv8s) and +0.7 (YOLOv8n) points of mAP@0.5:0.95.
  It improves five of six classes in both models and is slightly worse on CARDBOARD (-0.008).
- YOLOv8s beats YOLOv8n in all six classes under both recipes.

### Recommended model: YOLOv8s + `proposed`

Weights: `reports/runs/yolov8s_proposed/weights/best.pt`. It has the best mAP@0.5:0.95 and F1 of
the four trained models and runs in real time on a T4 GPU (55 FPS). If the demo has no GPU, use
`reports/runs/yolov8n_proposed/weights/best.pt` (about 2.6 times faster on CPU, 2.4 points lower
mAP@0.5:0.95).

```python
from ultralytics import YOLO

model = YOLO("reports/runs/yolov8s_proposed/weights/best.pt")
results = model.predict("image.jpg", imgsz=640, conf=0.25)  # also accepts a video path
print(model.names)  # BIODEGRADABLE, CARDBOARD, GLASS, METAL, PAPER, PLASTIC
```

`BoxF1_curve.png` in each run folder shows the F1 for every confidence threshold, useful to tune
`conf`. Expect the weakest detections on BIODEGRADABLE, CARDBOARD and PAPER (mAP@0.5:0.95 of
0.36 to 0.43 in YOLOv8s).

### Limitations

- Each model and recipe was trained once, so the differences are not tested for statistical
  significance.
- The `ultralytics_defaults` runs used ultralytics 8.4.173 and the `proposed` runs 8.4.172.
- Only two of the four recipes were trained.
- Confusion matrices are computed on the validation split, not on test.

## Development

```bash
uv run pytest         # tests
uv run ruff check .   # lint
uv run ruff format .  # format
```

Notebooks can import helpers directly (editable install):

```python
from vpc2.data import io
from vpc2.data.processing import run_pipeline
```

## Project structure

```
.
├── configs/
│   └── augmentation.yaml          # Augmentation policy (arms of the ablation)
├── data/                          # Not versioned
│   ├── raw/                       # Roboflow export (immutable)
│   ├── interim/                   # Audit reports
│   ├── processed/                 # Split ready for training
│   └── augmented/                 # Preview sample of the policy (not for training)
├── notebooks/
│   ├── 01_data_processing.ipynb
│   ├── 01_exploration.ipynb
│   ├── 02_eda.ipynb
│   ├── 03_augmentation_analysis.ipynb
│   └── 04_training_colab.ipynb          # Training + evaluation on a Colab GPU
├── scripts/
│   ├── process_data.py            # CLI for the data pipeline
│   └── train_eval.py              # Fine-tune YOLO, evaluate on test, measure latency
├── src/vpc2/
│   ├── data/
│   │   ├── augmentation.py        # Load the policy, preview/export the train split
│   │   ├── io.py                  # List/load images, YAML, discover raw
│   │   └── processing.py          # Audit, clean labels, split, write
│   └── utils/
├── tests/
├── reports/                       # Literature notes for the paper
│   └── runs/                      # Training results: metrics, weights, curves (see its README)
├── pyproject.toml
└── README.md
```
