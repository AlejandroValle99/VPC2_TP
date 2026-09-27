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
model.train(data="data/processed/data.yaml", **policy.train_kwargs())  # once ultralytics is added
```

Name the baseline arm "Ultralytics defaults" (Ultralytics augments by default), not "no
augmentation".

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
│   └── 03_augmentation_analysis.ipynb
├── scripts/
│   └── process_data.py            # CLI for the data pipeline
├── src/vpc2/
│   ├── data/
│   │   ├── augmentation.py        # Load the policy, preview/export the train split
│   │   ├── io.py                  # List/load images, YAML, discover raw
│   │   └── processing.py          # Audit, clean labels, split, write
│   └── utils/
├── tests/
├── reports/                       # Literature notes for the paper
├── pyproject.toml
└── README.md
```
