# VPC2

Computer vision experiments.

> Project description

The work happens in `notebooks/`. `src/vpc2/` holds the small amount of code that
is worth sharing between them.

## Requirements

- Python 3.12+
- [uv](https://docs.astral.sh/uv/)

## Setup

```bash
uv sync
```

This creates `.venv/` and installs the project along with its dependencies.

## Usage

```bash
uv run jupyter lab
```

Because the project is installed in editable mode, notebooks can import the
helpers directly:

```python
from vpc2.data import io
from vpc2.utils import visualization as viz
```

Put the source images under `data/raw/` — the directory is ignored by git.

## Development

```bash
uv sync --group dev   # add the dev tools
uv run pytest         # tests
uv run ruff check .   # lint
uv run ruff format .  # format
```

## Project structure

```
.
├── data/                        # Datasets (not versioned)
│   ├── raw/                     # Original, immutable data
│   ├── interim/                 # Intermediate transformations
│   └── processed/               # Data ready to experiment with
├── notebooks/                   # The experiments
│   └── 01_exploration.ipynb
├── src/vpc2/                    # Helpers shared between notebooks
│   ├── data/
│   │   └── io.py                # Listing and loading images
│   └── utils/
│       ├── __init__.py          # Seeding
│       ├── metrics.py           # Evaluation metrics
│       └── visualization.py     # Image plotting
├── tests/
├── pyproject.toml               # Project metadata and dependencies
└── README.md
```

## Notes
