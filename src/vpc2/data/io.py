from __future__ import annotations

from collections.abc import Iterable, Sequence
from pathlib import Path

import numpy as np
import yaml
from numpy.typing import NDArray
from PIL import Image, UnidentifiedImageError

IMAGE_SUFFIXES: tuple[str, ...] = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")
LABEL_SUFFIX: str = ".txt"
README_TXT_PREFIXES: tuple[str, ...] = ("readme",)

ArrayRgb = NDArray[np.uint8]


def _as_path(path: str | Path) -> Path:
    return Path(path)


def list_images(
    directory: str | Path,
    suffixes: Sequence[str] = IMAGE_SUFFIXES,
    *,
    recursive: bool = False,
) -> list[Path]:
    root = _as_path(directory)
    if not root.is_dir():
        return []

    allowed = {suffix.lower() for suffix in suffixes}
    iterator: Iterable[Path] = root.rglob("*") if recursive else root.iterdir()
    images = [
        path
        for path in iterator
        if path.is_file() and path.suffix.lower() in allowed
    ]
    return sorted(images)


def list_label_files(
    directory: str | Path,
    *,
    recursive: bool = False,
) -> list[Path]:
    root = _as_path(directory)
    if not root.is_dir():
        return []

    iterator: Iterable[Path] = root.rglob("*.txt") if recursive else root.glob("*.txt")
    labels: list[Path] = []
    for path in iterator:
        if not path.is_file():
            continue
        if path.name.lower().startswith(README_TXT_PREFIXES):
            continue
        labels.append(path)
    return sorted(labels)


def load_image(path: str | Path, mode: str = "RGB") -> ArrayRgb:
    image_path = _as_path(path)
    if not image_path.is_file():
        raise FileNotFoundError(image_path)

    requested = mode.upper()
    if requested not in {"RGB", "BGR"}:
        raise ValueError(f"Unsupported image mode: {mode!r}")

    try:
        with Image.open(image_path) as image:
            image.load()
            rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
    except (OSError, UnidentifiedImageError, ValueError) as exc:
        raise ValueError(f"Unreadable image: {image_path}") from exc

    if rgb.ndim != 3 or rgb.shape[2] != 3 or rgb.size == 0:
        raise ValueError(f"Invalid RGB array from {image_path}: shape={rgb.shape}")

    if requested == "BGR":
        return np.ascontiguousarray(rgb[:, :, ::-1])
    return np.ascontiguousarray(rgb)


def load_images(paths: Sequence[str | Path], mode: str = "RGB") -> list[ArrayRgb]:
    return [load_image(path, mode=mode) for path in paths]


def stem_of(path: str | Path) -> str:
    return _as_path(path).stem


def matching_label_path(image_path: str | Path, labels_dir: str | Path) -> Path:
    return _as_path(labels_dir) / f"{stem_of(image_path)}{LABEL_SUFFIX}"


def load_yolo_yaml(path: str | Path) -> dict:
    yaml_path = _as_path(path)
    if not yaml_path.is_file():
        return {}
    with yaml_path.open(encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    if not isinstance(payload, dict):
        raise TypeError(f"Expected a mapping in {yaml_path}")
    return payload


def class_names_from_yaml(payload: dict) -> list[str]:
    names = payload.get("names", [])
    if isinstance(names, dict):
        return [str(names[key]) for key in sorted(names, key=lambda item: int(item))]
    if isinstance(names, list):
        return [str(name) for name in names]
    return []


def write_yolo_yaml(path: str | Path, payload: dict) -> Path:
    yaml_path = _as_path(path)
    yaml_path.parent.mkdir(parents=True, exist_ok=True)
    with yaml_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(payload, handle, sort_keys=False, allow_unicode=True)
    return yaml_path


def discover_raw_dataset(raw_root: str | Path) -> Path:
    """Find the YOLO root under `raw_root`: garbage-classification-3/, then raw_root, then the first nested data.yaml."""
    root = _as_path(raw_root)
    if not root.is_dir():
        raise FileNotFoundError(f"Raw data directory not found: {root}")

    preferred = root / "garbage-classification-3"
    if (preferred / "data.yaml").is_file():
        return preferred.resolve()
    if (root / "data.yaml").is_file():
        return root.resolve()

    matches = sorted(path.parent.resolve() for path in root.rglob("data.yaml"))
    if not matches:
        raise FileNotFoundError(
            f"No data.yaml found under {root}. Extract the Roboflow YOLO zip into data/raw/."
        )
    return matches[0]
