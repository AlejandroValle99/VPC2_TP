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
├── data/                          # Not versioned
│   ├── raw/                       # Roboflow export (immutable)
│   ├── interim/                   # Audit reports
│   └── processed/                 # Split ready for training
├── notebooks/
│   ├── 01_data_processing.ipynb
│   └── 01_exploration.ipynb
├── scripts/
│   └── process_data.py            # CLI for the data pipeline
├── src/vpc2/
│   ├── data/
│   │   ├── io.py                  # List/load images, YAML, discover raw
│   │   └── processing.py          # Audit, clean labels, split, write
│   └── utils/
├── tests/
├── reports/                       # Literature notes for the paper
├── pyproject.toml
└── README.md
```
