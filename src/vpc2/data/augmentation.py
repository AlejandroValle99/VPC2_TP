"""Augmentation policy for training, plus a label-safe preview of it on the train split.

Training augments online: Ultralytics hyperparameters plus Albumentations objects, both taken from
``configs/augmentation.yaml``. ``export_preview`` writes a small offline sample to ``data/augmented/`` so
the policy can be inspected. It reads only ``images/train`` and never writes into raw/ or processed/.
"""

from __future__ import annotations

import json
import shutil
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import albumentations as A
import cv2
import numpy as np
import yaml
from PIL import Image

from vpc2.data import io
from vpc2.data.processing import YoloBox, parse_yolo_label

DEFAULT_POLICY_PATH = Path(__file__).resolve().parents[3] / "configs" / "augmentation.yaml"

# Ultralytics augmentation parameters an arm may override (cfg/default.yaml on main, checked 2026-09-23).
ULTRALYTICS_AUGMENTATION_KEYS = frozenset(
    {
        "hsv_h",
        "hsv_s",
        "hsv_v",
        "degrees",
        "translate",
        "scale",
        "shear",
        "perspective",
        "flipud",
        "fliplr",
        "bgr",
        "mosaic",
        "mixup",
        "cutmix",
        "copy_paste",
        "close_mosaic",
    }
)
# Ultralytics defaults for the two flip parameters the preview mimics when an arm does not override them.
ULTRALYTICS_PREVIEW_DEFAULTS = {"fliplr": 0.5, "flipud": 0.0}

# Chart palette slots 1-6, one per class in data.yaml order (BIODEGRADABLE ... PLASTIC).
CLASS_COLORS_RGB = (
    (0x2A, 0x78, 0xD6),
    (0xEB, 0x68, 0x34),
    (0x1B, 0xAF, 0x7A),
    (0xED, 0xA1, 0x00),
    (0xE8, 0x7B, 0xA4),
    (0x00, 0x83, 0x00),
)
SHEET_ROWS = 6
SHEET_TILE = 192


@dataclass(frozen=True, slots=True)
class AugmentationPolicy:
    name: str
    description: str
    ultralytics: dict[str, float]
    custom: tuple[dict[str, Any], ...] = ()

    def build_custom(self) -> list[A.BasicTransform]:
        transforms: list[A.BasicTransform] = []
        for spec in self.custom:
            transform_cls = getattr(A, spec["name"], None)
            if transform_cls is None:
                raise ValueError(
                    f"Arm {self.name!r}: unknown Albumentations transform {spec['name']!r}"
                )
            params = {
                key: tuple(value) if isinstance(value, list) else value
                for key, value in (spec.get("params") or {}).items()
            }
            transforms.append(transform_cls(**params))
        return transforms

    def train_kwargs(self) -> dict[str, Any]:
        """Keyword arguments for ``model.train(...)``: overrides plus the custom transforms."""
        kwargs: dict[str, Any] = dict(self.ultralytics)
        custom = self.build_custom()
        if custom:
            kwargs["augmentations"] = custom
        return kwargs

    def setting(self, key: str) -> float:
        """Value the run will use for a preview-mimicked parameter (override, else Ultralytics default)."""
        return float(self.ultralytics.get(key, ULTRALYTICS_PREVIEW_DEFAULTS[key]))

    def preview_pipeline(self, seed: int | None = None) -> A.Compose:
        """Flips (as configured) plus the custom transforms, with YOLO-format box handling."""
        flips = [
            A.HorizontalFlip(p=self.setting("fliplr")),
            A.VerticalFlip(p=self.setting("flipud")),
        ]
        return A.Compose(
            [*flips, *self.build_custom()],
            bbox_params=A.BboxParams(format="yolo", label_fields=["class_ids"], clip=True),
            seed=seed,
        )


def _read_arms(path: str | Path) -> dict[str, Any]:
    with Path(path).open(encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    arms = payload.get("arms")
    if not isinstance(arms, dict) or not arms:
        raise ValueError(f"{path}: expected a non-empty 'arms' mapping")
    return arms


def list_arms(path: str | Path | None = None) -> list[str]:
    return list(_read_arms(path or DEFAULT_POLICY_PATH))


def _resolve_arm(name: str, arms: dict[str, Any], chain: tuple[str, ...]) -> AugmentationPolicy:
    if name in chain:
        raise ValueError(f"Circular 'extends' in arms: {' -> '.join((*chain, name))}")
    if name not in arms:
        raise KeyError(f"Unknown arm {name!r}; available: {sorted(arms)}")
    spec = arms[name] or {}
    parent = _resolve_arm(spec["extends"], arms, (*chain, name)) if spec.get("extends") else None

    ultralytics = {**(parent.ultralytics if parent else {}), **(spec.get("ultralytics") or {})}
    unknown = set(ultralytics) - ULTRALYTICS_AUGMENTATION_KEYS
    if unknown:
        raise ValueError(f"Arm {name!r}: unknown Ultralytics parameters {sorted(unknown)}")
    custom = tuple(spec["custom"]) if "custom" in spec else (parent.custom if parent else ())
    description = " ".join(str(spec.get("description", "")).split())
    return AugmentationPolicy(name, description, ultralytics, custom)


def load_policy(arm: str, path: str | Path | None = None) -> AugmentationPolicy:
    return _resolve_arm(arm, _read_arms(path or DEFAULT_POLICY_PATH), ())


def augment_sample(
    policy: AugmentationPolicy, image: np.ndarray, boxes: np.ndarray, *, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    """Apply the preview pipeline to an RGB image and (n, 5) ``[class, xc, yc, w, h]`` boxes."""
    out = policy.preview_pipeline(seed=seed)(
        image=image,
        bboxes=boxes[:, 1:].tolist(),
        class_ids=boxes[:, 0].astype(int).tolist(),
    )
    new_boxes = np.array(out["bboxes"], dtype=float).reshape(-1, 4)
    class_ids = np.array(out["class_ids"], dtype=float).reshape(-1, 1)
    return out["image"], np.hstack([class_ids, new_boxes])


def simple_mosaic(
    samples: Sequence[tuple[np.ndarray, np.ndarray]], *, size: int = 416
) -> tuple[np.ndarray, np.ndarray]:
    """2x2 tiling of four samples, for illustration only (Ultralytics runs its own mosaic in training)."""
    if len(samples) != 4:
        raise ValueError(f"simple_mosaic needs exactly 4 samples, got {len(samples)}")
    half = size // 2
    canvas = np.zeros((size, size, 3), dtype=np.uint8)
    all_boxes: list[np.ndarray] = []
    for index, (image, boxes) in enumerate(samples):
        row, col = divmod(index, 2)
        tile = cv2.resize(image, (half, half), interpolation=cv2.INTER_AREA)
        canvas[row * half : (row + 1) * half, col * half : (col + 1) * half] = tile
        moved = boxes.astype(float).copy()
        moved[:, 1] = (col + moved[:, 1]) / 2.0
        moved[:, 2] = (row + moved[:, 2]) / 2.0
        moved[:, 3:] /= 2.0
        all_boxes.append(moved)
    return canvas, np.vstack(all_boxes)


def _load_boxes(label_path: Path, num_classes: int) -> np.ndarray:
    boxes, _, _ = parse_yolo_label(label_path, num_classes=num_classes)
    rows = [[b.class_id, b.x_center, b.y_center, b.width, b.height] for b in boxes]
    return np.array(rows, dtype=float).reshape(-1, 5)


def _overlaps(first: Path, second: Path) -> bool:
    first, second = first.resolve(), second.resolve()
    return first == second or first in second.parents or second in first.parents


def _draw_tile(image: np.ndarray, boxes: np.ndarray, caption: str) -> np.ndarray:
    tile = cv2.cvtColor(
        cv2.resize(image, (SHEET_TILE, SHEET_TILE), interpolation=cv2.INTER_AREA), cv2.COLOR_RGB2BGR
    )
    for class_id, xc, yc, w, h in boxes:
        x1, y1 = int((xc - w / 2) * SHEET_TILE), int((yc - h / 2) * SHEET_TILE)
        x2, y2 = int((xc + w / 2) * SHEET_TILE), int((yc + h / 2) * SHEET_TILE)
        red, green, blue = CLASS_COLORS_RGB[int(class_id) % len(CLASS_COLORS_RGB)]
        cv2.rectangle(tile, (x1, y1), (x2, y2), (blue, green, red), 1)
    cv2.rectangle(tile, (0, 0), (SHEET_TILE, 14), (0, 0, 0), -1)
    cv2.putText(
        tile, caption, (3, 11), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255, 255, 255), 1, cv2.LINE_AA
    )
    return tile


def _write_sample(image: np.ndarray, boxes: np.ndarray, image_path: Path, label_path: Path) -> None:
    image_path.parent.mkdir(parents=True, exist_ok=True)
    label_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(image).save(image_path, format="JPEG", quality=95, subsampling=0)
    lines = (YoloBox(int(c), xc, yc, w, h).as_line() for c, xc, yc, w, h in boxes)
    label_path.write_text("".join(f"{line}\n" for line in lines), encoding="utf-8")


@dataclass(frozen=True, slots=True)
class ExportSummary:
    out_dir: Path
    source_images: int
    files_written: int
    sheets: tuple[Path, ...]


def export_preview(
    policy: AugmentationPolicy,
    processed_dir: str | Path,
    out_dir: str | Path,
    *,
    per_class: int | None = 8,
    n_variants: int = 3,
    seed: int = 42,
    protected: Sequence[str | Path] = (),
) -> ExportSummary:
    """Write augmented copies of a stratified train sample plus contact sheets, for inspection only.

    Reads ``processed_dir/images/train`` and nothing else. ``per_class=None`` exports every train image.
    Refuses to run when ``out_dir`` overlaps ``processed_dir`` or any ``protected`` path (e.g. raw/).
    """
    processed, out = Path(processed_dir), Path(out_dir)
    for guarded in (processed, *map(Path, protected)):
        if _overlaps(out, guarded):
            raise ValueError(f"out_dir {out} overlaps protected path {guarded}")

    class_names = io.class_names_from_yaml(io.load_yolo_yaml(processed / "data.yaml"))
    labels_dir = processed / "labels" / "train"
    by_class: dict[int, list[tuple[Path, np.ndarray]]] = {}
    for image_path in io.list_images(processed / "images" / "train"):
        boxes = _load_boxes(io.matching_label_path(image_path, labels_dir), len(class_names))
        if len(boxes):
            majority = int(np.bincount(boxes[:, 0].astype(int)).argmax())
            by_class.setdefault(majority, []).append((image_path, boxes))

    rng = np.random.default_rng(seed)
    selected: dict[int, list[tuple[Path, np.ndarray]]] = {}
    for class_id, group in sorted(by_class.items()):
        if per_class is None or per_class >= len(group):
            selected[class_id] = group
        else:
            picks = sorted(rng.choice(len(group), size=per_class, replace=False))
            selected[class_id] = [group[i] for i in picks]

    for sub in ("images", "labels", "preview"):
        shutil.rmtree(out / sub, ignore_errors=True)
    out.mkdir(parents=True, exist_ok=True)

    files_written, sheets, sample_index = 0, [], 0
    for class_id, group in selected.items():
        rows: list[np.ndarray] = []
        for image_path, boxes in group:
            image = io.load_image(image_path)
            tiles = [_draw_tile(image, boxes, "original")]
            for variant in range(n_variants):
                variant_seed = int(
                    np.random.SeedSequence([seed, sample_index, variant]).generate_state(1)[0]
                )
                new_image, new_boxes = augment_sample(policy, image, boxes, seed=variant_seed)
                stem = f"{image_path.stem}__aug{variant}"
                _write_sample(
                    new_image,
                    new_boxes,
                    out / "images" / "train" / f"{stem}.jpg",
                    out / "labels" / "train" / f"{stem}.txt",
                )
                files_written += 1
                tiles.append(_draw_tile(new_image, new_boxes, f"aug {variant}"))
            rows.append(np.hstack(tiles))
            sample_index += 1
        sheet_path = out / "preview" / f"{class_names[class_id]}.jpg"
        sheet_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(sheet_path), np.vstack(rows[:SHEET_ROWS]), [cv2.IMWRITE_JPEG_QUALITY, 90])
        sheets.append(sheet_path)

    manifest = {
        "arm": policy.name,
        "ultralytics_overrides": policy.ultralytics,
        "custom_transforms": list(policy.custom),
        "seed": seed,
        "per_class": per_class,
        "n_variants": n_variants,
        "albumentations": A.__version__,
        "source_images": sum(len(group) for group in selected.values()),
        "files_written": files_written,
        "note": "Preview only. Training applies the policy online (see configs/augmentation.yaml).",
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return ExportSummary(out, manifest["source_images"], files_written, tuple(sheets))
