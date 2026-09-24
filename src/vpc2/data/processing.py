"""Audit, clean, and split a YOLO waste-detection dataset.

Raw files under ``data/raw/`` are never modified. Intermediate manifests land in
``data/interim/`` and the training-ready layout is written to ``data/processed/``.
"""

from __future__ import annotations

import json
import logging
import math
import random
import shutil
from collections import Counter, defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path

from PIL import Image, UnidentifiedImageError
from tqdm import tqdm

from vpc2.data import io

LOGGER = logging.getLogger(__name__)

SPLIT_DIR_NAMES: tuple[str, ...] = ("train", "valid", "val", "test")
DEFAULT_RATIOS: tuple[float, float, float] = (0.70, 0.20, 0.10)
MIN_BOX_SIZE = 1e-6
COORD_DECIMALS = 6


@dataclass(frozen=True, slots=True)
class YoloBox:
    class_id: int
    x_center: float
    y_center: float
    width: float
    height: float

    def as_line(self) -> str:
        return (
            f"{self.class_id} {self.x_center:.{COORD_DECIMALS}f} "
            f"{self.y_center:.{COORD_DECIMALS}f} {self.width:.{COORD_DECIMALS}f} "
            f"{self.height:.{COORD_DECIMALS}f}"
        )

    def key(self) -> tuple[int, float, float, float, float]:
        return (
            self.class_id,
            round(self.x_center, COORD_DECIMALS),
            round(self.y_center, COORD_DECIMALS),
            round(self.width, COORD_DECIMALS),
            round(self.height, COORD_DECIMALS),
        )


@dataclass(slots=True)
class PairedSample:
    stem: str
    image_path: Path
    label_path: Path
    boxes: list[YoloBox]
    converted: bool
    issues: list[str] = field(default_factory=list)

    @property
    def majority_class(self) -> int:
        counts = Counter(box.class_id for box in self.boxes)
        return counts.most_common(1)[0][0]


@dataclass(slots=True)
class PipelineReport:
    dataset_root: str
    class_names: list[str]
    images_scanned: int = 0
    images_valid: int = 0
    images_corrupt: int = 0
    images_converted: int = 0
    labels_scanned: int = 0
    orphan_images: int = 0
    orphan_labels: int = 0
    empty_labels: int = 0
    duplicate_stems: int = 0
    boxes_raw: int = 0
    boxes_kept: int = 0
    boxes_clipped: int = 0
    boxes_dropped: int = 0
    duplicates_removed: int = 0
    malformed_lines: int = 0
    invalid_class: int = 0
    paired_kept: int = 0
    split_train: int = 0
    split_val: int = 0
    split_test: int = 0
    class_histogram: dict[str, int] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)

    def log_summary(self) -> None:
        LOGGER.info("Dataset root: %s", self.dataset_root)
        LOGGER.info("Classes (%d): %s", len(self.class_names), self.class_names)
        LOGGER.info(
            "Images scanned=%d valid=%d corrupt=%d converted=%d",
            self.images_scanned,
            self.images_valid,
            self.images_corrupt,
            self.images_converted,
        )
        LOGGER.info(
            "Labels scanned=%d orphan_images=%d orphan_labels=%d empty=%d",
            self.labels_scanned,
            self.orphan_images,
            self.orphan_labels,
            self.empty_labels,
        )
        LOGGER.info(
            "Boxes raw=%d kept=%d clipped=%d dropped=%d dups_removed=%d malformed=%d invalid_class=%d",
            self.boxes_raw,
            self.boxes_kept,
            self.boxes_clipped,
            self.boxes_dropped,
            self.duplicates_removed,
            self.malformed_lines,
            self.invalid_class,
        )
        LOGGER.info(
            "Paired samples kept=%d | split train/val/test = %d/%d/%d",
            self.paired_kept,
            self.split_train,
            self.split_val,
            self.split_test,
        )
        if self.class_histogram:
            LOGGER.info("Class histogram (instances): %s", self.class_histogram)
        for warning in self.warnings:
            LOGGER.warning(warning)


def collect_yolo_files(dataset_root: str | Path) -> tuple[list[Path], list[Path]]:
    root = Path(dataset_root)
    images: list[Path] = []
    labels: list[Path] = []

    for split in SPLIT_DIR_NAMES:
        images.extend(io.list_images(root / split / "images"))
        labels.extend(io.list_label_files(root / split / "labels"))
        images.extend(io.list_images(root / "images" / split))
        labels.extend(io.list_label_files(root / "labels" / split))

    if not images:
        images.extend(io.list_images(root / "images"))
        labels.extend(io.list_label_files(root / "labels"))
    if not images:
        images.extend(io.list_images(root, recursive=True))
        labels.extend(io.list_label_files(root / "labels", recursive=True))
        for split in SPLIT_DIR_NAMES:
            labels.extend(io.list_label_files(root / split / "labels", recursive=True))

    unique_images = sorted(set(images))
    unique_labels = sorted(set(labels))
    return unique_images, unique_labels


def audit_image(path: str | Path) -> tuple[bool, bool, str | None]:
    image_path = Path(path)
    try:
        size = image_path.stat().st_size
    except OSError as exc:
        return False, False, f"stat_failed: {exc}"
    if size <= 0:
        return False, False, "zero_byte"

    try:
        with Image.open(image_path) as image:
            image.load()
            width, height = image.size
            if width <= 0 or height <= 0:
                return False, False, "empty_dimensions"
            needs_conversion = image.mode != "RGB"
            if image.mode == "RGB" and image.getbands() != ("R", "G", "B"):
                needs_conversion = True
    except (OSError, UnidentifiedImageError, ValueError, SyntaxError) as exc:
        return False, False, f"unreadable: {exc}"
    return True, needs_conversion, None


def _box_extent(box: YoloBox) -> tuple[float, float, float, float]:
    return (
        box.x_center - box.width / 2.0,
        box.y_center - box.height / 2.0,
        box.x_center + box.width / 2.0,
        box.y_center + box.height / 2.0,
    )


def _clip_box(box: YoloBox) -> tuple[YoloBox | None, bool]:
    x1, y1, x2, y2 = _box_extent(box)

    clipped = False
    if min(x1, y1, x2, y2) < 0.0 or max(x1, y1, x2, y2) > 1.0:
        clipped = True

    x1 = min(max(x1, 0.0), 1.0)
    y1 = min(max(y1, 0.0), 1.0)
    x2 = min(max(x2, 0.0), 1.0)
    y2 = min(max(y2, 0.0), 1.0)
    width = x2 - x1
    height = y2 - y1
    if width < MIN_BOX_SIZE or height < MIN_BOX_SIZE:
        return None, True

    return (
        YoloBox(
            class_id=box.class_id,
            x_center=x1 + width / 2.0,
            y_center=y1 + height / 2.0,
            width=width,
            height=height,
        ),
        clipped,
    )


def parse_yolo_label(
    path: str | Path,
    *,
    num_classes: int,
    clip_out_of_bounds: bool = True,
) -> tuple[list[YoloBox], dict[str, int], list[str]]:
    label_path = Path(path)
    counters = {
        "boxes_raw": 0,
        "boxes_kept": 0,
        "boxes_clipped": 0,
        "boxes_dropped": 0,
        "duplicates_removed": 0,
        "malformed_lines": 0,
        "invalid_class": 0,
    }
    issues: list[str] = []
    kept: list[YoloBox] = []
    seen: set[tuple[int, float, float, float, float]] = set()

    try:
        text = label_path.read_text(encoding="utf-8")
    except OSError as exc:
        issues.append(f"{label_path.name}: unreadable_label ({exc})")
        return [], counters, issues

    if not text.strip():
        issues.append(f"{label_path.name}: empty_label")
        return [], counters, issues

    for line_number, raw_line in enumerate(text.splitlines(), start=1):
        line = raw_line.strip()
        if not line:
            continue
        counters["boxes_raw"] += 1
        parts = line.split()
        if len(parts) != 5:
            counters["malformed_lines"] += 1
            counters["boxes_dropped"] += 1
            issues.append(f"{label_path.name}:{line_number}: expected 5 fields, got {len(parts)}")
            continue
        try:
            class_raw = float(parts[0])
            if not class_raw.is_integer():
                raise ValueError("class_id is not an integer")
            class_id = int(class_raw)
            coords = tuple(float(value) for value in parts[1:])
        except ValueError:
            counters["malformed_lines"] += 1
            counters["boxes_dropped"] += 1
            issues.append(f"{label_path.name}:{line_number}: malformed numbers")
            continue

        if any(not math.isfinite(coord) for coord in coords):
            counters["malformed_lines"] += 1
            counters["boxes_dropped"] += 1
            issues.append(f"{label_path.name}:{line_number}: non-finite coordinate")
            continue

        if class_id < 0 or class_id >= num_classes:
            counters["invalid_class"] += 1
            counters["boxes_dropped"] += 1
            issues.append(
                f"{label_path.name}:{line_number}: class_id {class_id} outside [0, {num_classes})"
            )
            continue

        box = YoloBox(class_id, *coords)
        if clip_out_of_bounds:
            clipped_box, was_clipped = _clip_box(box)
            if clipped_box is None:
                counters["boxes_dropped"] += 1
                issues.append(f"{label_path.name}:{line_number}: box collapsed after clip")
                continue
            if was_clipped:
                counters["boxes_clipped"] += 1
                issues.append(f"{label_path.name}:{line_number}: clipped to [0, 1]")
            box = clipped_box
        else:
            x1, y1, x2, y2 = _box_extent(box)
            if min(x1, y1, x2, y2) < 0.0 or max(x1, y1, x2, y2) > 1.0:
                counters["boxes_dropped"] += 1
                issues.append(f"{label_path.name}:{line_number}: out of bounds")
                continue
            if box.width < MIN_BOX_SIZE or box.height < MIN_BOX_SIZE:
                counters["boxes_dropped"] += 1
                issues.append(f"{label_path.name}:{line_number}: degenerate box")
                continue

        key = box.key()
        if key in seen:
            counters["duplicates_removed"] += 1
            issues.append(f"{label_path.name}:{line_number}: duplicate box")
            continue
        seen.add(key)
        kept.append(box)
        counters["boxes_kept"] += 1

    return kept, counters, issues


def _index_by_stem(paths: Sequence[Path]) -> dict[str, list[Path]]:
    index: dict[str, list[Path]] = defaultdict(list)
    for path in paths:
        index[io.stem_of(path)].append(path)
    return index


def pair_and_clean(
    images: Sequence[Path],
    labels: Sequence[Path],
    *,
    num_classes: int,
    clip_out_of_bounds: bool = True,
) -> tuple[list[PairedSample], PipelineReport]:
    report = PipelineReport(dataset_root="", class_names=[])
    image_index = _index_by_stem(images)
    label_index = _index_by_stem(labels)
    report.images_scanned = len(images)
    report.labels_scanned = len(labels)

    for stem, paths in image_index.items():
        if len(paths) > 1:
            report.duplicate_stems += 1
            report.warnings.append(f"Duplicate image stem {stem!r}: {[str(p) for p in paths]}")
    for stem, paths in label_index.items():
        if len(paths) > 1:
            report.duplicate_stems += 1
            report.warnings.append(f"Duplicate label stem {stem!r}: {[str(p) for p in paths]}")

    report.orphan_images = sum(1 for stem in image_index if stem not in label_index)

    valid_images: dict[str, tuple[Path, bool]] = {}
    for stem, image_paths in tqdm(sorted(image_index.items()), desc="Audit images", unit="img"):
        image_path = image_paths[0]
        is_valid, needs_conversion, error = audit_image(image_path)
        if not is_valid:
            report.images_corrupt += 1
            report.warnings.append(f"Corrupt image {image_path}: {error}")
            continue
        report.images_valid += 1
        if needs_conversion:
            report.images_converted += 1
        valid_images[stem] = (image_path, needs_conversion)

    for stem in label_index:
        if stem in valid_images or stem in image_index:
            continue
        report.orphan_labels += 1

    paired: list[PairedSample] = []
    for stem, (image_path, needs_conversion) in tqdm(
        sorted(valid_images.items()), desc="Clean labels", unit="lbl"
    ):
        label_paths = label_index.get(stem, [])
        if not label_paths:
            continue

        label_path = label_paths[0]
        boxes, counters, issues = parse_yolo_label(
            label_path,
            num_classes=num_classes,
            clip_out_of_bounds=clip_out_of_bounds,
        )
        for key, value in counters.items():
            setattr(report, key, getattr(report, key) + value)

        if not boxes:
            report.empty_labels += 1
            continue

        paired.append(
            PairedSample(
                stem=stem,
                image_path=image_path,
                label_path=label_path,
                boxes=boxes,
                converted=needs_conversion,
                issues=issues,
            )
        )

    report.paired_kept = len(paired)
    return paired, report


def stratified_split(
    samples: Sequence[PairedSample],
    *,
    ratios: tuple[float, float, float] = DEFAULT_RATIOS,
    seed: int = 42,
) -> dict[str, list[PairedSample]]:
    train_ratio, val_ratio, test_ratio = ratios
    total_ratio = train_ratio + val_ratio + test_ratio
    if abs(total_ratio - 1.0) > 1e-6:
        raise ValueError(f"Split ratios must sum to 1.0, got {ratios}")
    if any(ratio < 0 for ratio in ratios):
        raise ValueError(f"Split ratios must be non-negative, got {ratios}")

    rng = random.Random(seed)
    by_class: dict[int, list[PairedSample]] = defaultdict(list)
    for sample in samples:
        by_class[sample.majority_class].append(sample)

    splits: dict[str, list[PairedSample]] = {"train": [], "val": [], "test": []}
    for class_id in sorted(by_class):
        group = list(by_class[class_id])
        rng.shuffle(group)
        n = len(group)
        n_test = round(n * test_ratio)
        n_val = round(n * val_ratio)
        if n > 0 and n_test + n_val >= n:
            n_test = min(n_test, max(0, n - 1))
            n_val = min(n_val, max(0, n - 1 - n_test))
        n_train = n - n_val - n_test
        splits["train"].extend(group[:n_train])
        splits["val"].extend(group[n_train : n_train + n_val])
        splits["test"].extend(group[n_train + n_val :])

    for split_samples in splits.values():
        split_samples.sort(key=lambda sample: sample.stem)
    return splits


def _write_label(path: Path, boxes: Sequence[YoloBox]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(f"{box.as_line()}\n" for box in boxes), encoding="utf-8")


def _materialize_rgb_image(source: Path, destination: Path, *, convert: bool) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if not convert and source.suffix.lower() in {".jpg", ".jpeg"}:
        shutil.copy2(source, destination)
        return
    with Image.open(source) as image:
        image.load()
        image.convert("RGB").save(destination, format="JPEG", quality=95, subsampling=0)


def write_processed_dataset(
    splits: dict[str, Sequence[PairedSample]],
    processed_dir: str | Path,
    *,
    class_names: Sequence[str],
) -> Path:
    root = Path(processed_dir)
    gitkeep = root / ".gitkeep"
    had_gitkeep = gitkeep.is_file()
    if root.exists():
        shutil.rmtree(root)
    root.mkdir(parents=True, exist_ok=True)
    if had_gitkeep:
        gitkeep.touch()

    for split_name, samples in splits.items():
        for sample in tqdm(samples, desc=f"Write {split_name}", unit="img"):
            image_dest = root / "images" / split_name / f"{sample.stem}.jpg"
            label_dest = root / "labels" / split_name / f"{sample.stem}.txt"
            _materialize_rgb_image(sample.image_path, image_dest, convert=sample.converted)
            _write_label(label_dest, sample.boxes)

    payload = {
        "path": root.resolve().as_posix(),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "nc": len(class_names),
        "names": {index: name for index, name in enumerate(class_names)},
    }
    return io.write_yolo_yaml(root / "data.yaml", payload)


def _write_lines(path: Path, lines: Iterable[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def write_interim_artifacts(
    *,
    interim_dir: str | Path,
    report: PipelineReport,
    images: Sequence[Path],
    labels: Sequence[Path],
    paired: Sequence[PairedSample],
    issues: Sequence[str],
) -> None:
    interim = Path(interim_dir)
    audit_dir = interim / "audit"
    orphan_dir = interim / "orphans"
    audit_dir.mkdir(parents=True, exist_ok=True)
    orphan_dir.mkdir(parents=True, exist_ok=True)

    image_stems = {io.stem_of(path) for path in images}
    label_stems = {io.stem_of(path) for path in labels}
    paired_stems = {sample.stem for sample in paired}

    orphan_images = sorted(str(path) for path in images if io.stem_of(path) not in label_stems)
    orphan_labels = sorted(str(path) for path in labels if io.stem_of(path) not in image_stems)
    dropped_empty = sorted(
        str(path)
        for path in labels
        if io.stem_of(path) in image_stems and io.stem_of(path) not in paired_stems
    )

    _write_lines(orphan_dir / "orphan_images.txt", orphan_images)
    _write_lines(orphan_dir / "orphan_labels.txt", orphan_labels)
    _write_lines(audit_dir / "dropped_or_empty.txt", dropped_empty)
    _write_lines(audit_dir / "annotation_issues.txt", issues)

    report_path = audit_dir / "report.json"
    report_path.write_text(json.dumps(asdict(report), indent=2), encoding="utf-8")


def _class_histogram(samples: Sequence[PairedSample], class_names: Sequence[str]) -> dict[str, int]:
    counts: Counter[int] = Counter()
    for sample in samples:
        counts.update(box.class_id for box in sample.boxes)
    histogram: dict[str, int] = {}
    for index, name in enumerate(class_names):
        histogram[name] = int(counts.get(index, 0))
    return histogram


def run_pipeline(
    raw_dir: str | Path = "data/raw",
    interim_dir: str | Path = "data/interim",
    processed_dir: str | Path = "data/processed",
    *,
    seed: int = 42,
    ratios: tuple[float, float, float] = DEFAULT_RATIOS,
    clip_out_of_bounds: bool = True,
) -> PipelineReport:
    """Audit, clean, and split. Does not modify data/raw/."""
    dataset_root = io.discover_raw_dataset(raw_dir)
    LOGGER.info("Using raw dataset at %s", dataset_root)

    yaml_payload = io.load_yolo_yaml(dataset_root / "data.yaml")
    class_names = io.class_names_from_yaml(yaml_payload)
    images, labels = collect_yolo_files(dataset_root)

    if not class_names:
        max_class = -1
        for label_path in labels:
            boxes, _, _ = parse_yolo_label(label_path, num_classes=10_000, clip_out_of_bounds=True)
            if boxes:
                max_class = max(max_class, max(box.class_id for box in boxes))
        class_names = [f"class_{index}" for index in range(max_class + 1)]
        LOGGER.warning("data.yaml had no names; inferred %d classes", len(class_names))

    paired, report = pair_and_clean(
        images,
        labels,
        num_classes=len(class_names),
        clip_out_of_bounds=clip_out_of_bounds,
    )
    report.dataset_root = str(dataset_root)
    report.class_names = list(class_names)
    report.class_histogram = _class_histogram(paired, class_names)

    expected = 10464
    if report.images_scanned < expected:
        report.warnings.append(
            f"Raw export looks incomplete: scanned {report.images_scanned} images, "
            f"Roboflow v2 lists {expected}. Re-extract the zip if train/valid are missing."
        )
    if report.orphan_images:
        report.warnings.append(
            f"{report.orphan_images} images have no matching .txt label "
            "(often a partial unzip: train/labels missing)."
        )

    if not paired:
        write_interim_artifacts(
            interim_dir=interim_dir,
            report=report,
            images=images,
            labels=labels,
            paired=paired,
            issues=[],
        )
        report.log_summary()
        raise RuntimeError(
            "No paired image/label samples survived the audit. "
            "Check data/interim/audit/report.json and re-extract the Roboflow zip."
        )

    splits = stratified_split(paired, ratios=ratios, seed=seed)
    report.split_train = len(splits["train"])
    report.split_val = len(splits["val"])
    report.split_test = len(splits["test"])

    all_issues = [issue for sample in paired for issue in sample.issues]
    write_interim_artifacts(
        interim_dir=interim_dir,
        report=report,
        images=images,
        labels=labels,
        paired=paired,
        issues=all_issues,
    )
    write_processed_dataset(splits, processed_dir, class_names=class_names)
    report.log_summary()
    LOGGER.info("Processed dataset written to %s", Path(processed_dir).resolve())
    LOGGER.info("Audit report written to %s", Path(interim_dir) / "audit" / "report.json")
    return report
