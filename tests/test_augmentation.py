import hashlib
from pathlib import Path

import albumentations as A
import numpy as np
import pytest
from PIL import Image

from vpc2.data.augmentation import (
    ULTRALYTICS_AUGMENTATION_KEYS,
    AugmentationPolicy,
    augment_sample,
    export_preview,
    list_arms,
    load_policy,
    simple_mosaic,
)

BOX = np.array([[1.0, 0.25, 0.30, 0.20, 0.10]])


def _noise_image(seed: int = 0, size: int = 64) -> np.ndarray:
    return np.random.default_rng(seed).integers(0, 255, (size, size, 3), dtype=np.uint8)


def _policy(ultralytics: dict | None = None, custom: tuple = ()) -> AugmentationPolicy:
    return AugmentationPolicy("test", "", ultralytics or {}, custom)


def test_default_policy_arms_load_and_build() -> None:
    required = {"no_augmentation", "ultralytics_defaults", "proposed", "proposed_low_saturation"}
    assert required <= set(list_arms())
    for arm in list_arms():
        load_policy(arm).train_kwargs()


def test_no_augmentation_zeroes_every_ultralytics_parameter() -> None:
    overrides = load_policy("no_augmentation").ultralytics
    assert set(overrides) == ULTRALYTICS_AUGMENTATION_KEYS - {"close_mosaic"}
    assert all(value == 0.0 for value in overrides.values())


def test_ultralytics_defaults_arm_passes_nothing() -> None:
    assert load_policy("ultralytics_defaults").train_kwargs() == {}


def test_extending_arm_merges_overrides_and_inherits_custom_transforms() -> None:
    proposed = load_policy("proposed")
    child = load_policy("proposed_low_saturation")
    assert child.ultralytics == {**proposed.ultralytics, "hsv_s": 0.5}
    assert child.custom == proposed.custom
    kwargs = child.train_kwargs()
    assert kwargs["flipud"] == 0.5 and kwargs["hsv_s"] == 0.5
    assert len(kwargs["augmentations"]) == len(proposed.custom) == 3


def test_load_policy_rejects_bad_config(tmp_path: Path) -> None:
    config = tmp_path / "policy.yaml"
    config.write_text(
        "arms:\n  typo:\n    ultralytics: {hsv_ss: 0.5}\n"
        "  a:\n    extends: b\n  b:\n    extends: a\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="hsv_ss"):
        load_policy("typo", config)
    with pytest.raises(ValueError, match="Circular"):
        load_policy("a", config)
    with pytest.raises(KeyError, match="missing"):
        load_policy("missing", config)


def test_unknown_custom_transform_is_rejected() -> None:
    policy = _policy(custom=({"name": "NotATransform", "params": {}},))
    with pytest.raises(ValueError, match="NotATransform"):
        policy.build_custom()


def test_flips_move_boxes_and_image_together() -> None:
    image = _noise_image()
    flipped, boxes = augment_sample(_policy({"fliplr": 1.0, "flipud": 0.0}), image, BOX, seed=0)
    assert np.array_equal(flipped, image[:, ::-1])
    assert np.allclose(boxes, [[1.0, 0.75, 0.30, 0.20, 0.10]])

    flipped, boxes = augment_sample(_policy({"fliplr": 0.0, "flipud": 1.0}), image, BOX, seed=0)
    assert np.array_equal(flipped, image[::-1])
    assert np.allclose(boxes, [[1.0, 0.25, 0.70, 0.20, 0.10]])


def test_custom_transforms_change_pixels_but_not_labels() -> None:
    custom = (
        {
            "name": "RandomBrightnessContrast",
            "params": {"brightness_limit": [-0.4, -0.4], "contrast_limit": [0.0, 0.0], "p": 1.0},
        },
        {"name": "MotionBlur", "params": {"blur_limit": [5, 5], "p": 1.0}},
    )
    policy = _policy({"fliplr": 0.0, "flipud": 0.0}, custom)
    image = _noise_image()
    out, boxes = augment_sample(policy, image, BOX, seed=3)
    assert np.allclose(boxes, BOX)
    assert out.shape == image.shape and out.dtype == np.uint8
    assert out.mean() < image.mean()


def test_same_seed_reproduces_and_different_seed_differs() -> None:
    custom = ({"name": "MotionBlur", "params": {"blur_limit": [3, 9], "p": 1.0}},)
    policy = _policy({"fliplr": 0.0, "flipud": 0.0}, custom)
    image = _noise_image()
    first, _ = augment_sample(policy, image, BOX, seed=1)
    again, _ = augment_sample(policy, image, BOX, seed=1)
    other, _ = augment_sample(policy, image, BOX, seed=2)
    assert np.array_equal(first, again)
    assert not np.array_equal(first, other)


def test_random_rotate90_moves_boxes_exactly() -> None:
    expected = {
        (0.25, 0.30, 0.20, 0.10),
        (0.30, 0.75, 0.10, 0.20),
        (0.75, 0.70, 0.20, 0.10),
        (0.70, 0.25, 0.10, 0.20),
    }
    seen = set()
    for seed in range(40):
        pipeline = A.Compose(
            [A.RandomRotate90(p=1.0)],
            bbox_params=A.BboxParams(format="yolo", label_fields=["class_ids"]),
            seed=seed,
        )
        out = pipeline(image=_noise_image(), bboxes=[tuple(BOX[0, 1:])], class_ids=[1])
        seen.add(tuple(round(float(v), 4) for v in out["bboxes"][0]))
    assert seen == expected


def test_simple_mosaic_places_boxes_in_their_tile() -> None:
    samples = [(_noise_image(i), np.array([[i, 0.5, 0.5, 0.2, 0.2]])) for i in range(4)]
    canvas, boxes = simple_mosaic(samples, size=64)
    assert canvas.shape == (64, 64, 3)
    assert np.allclose(boxes[:, 1:3], [[0.25, 0.25], [0.75, 0.25], [0.25, 0.75], [0.75, 0.75]])
    assert np.allclose(boxes[:, 3:], 0.1)
    assert boxes[:, 0].tolist() == [0, 1, 2, 3]
    with pytest.raises(ValueError):
        simple_mosaic(samples[:3])


def _make_processed(root: Path, per_split: int = 4) -> Path:
    processed = root / "processed"
    (processed / "data.yaml").parent.mkdir(parents=True)
    (processed / "data.yaml").write_text("nc: 2\nnames: {0: paper, 1: glass}\n", encoding="utf-8")
    for split, prefix in (("train", "tr"), ("val", "va"), ("test", "te")):
        for index in range(per_split):
            stem = f"{prefix}_{index:02d}"
            image_path = processed / "images" / split / f"{stem}.jpg"
            label_path = processed / "labels" / split / f"{stem}.txt"
            image_path.parent.mkdir(parents=True, exist_ok=True)
            label_path.parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray(_noise_image(index)).save(image_path, format="JPEG")
            label_path.write_text(f"{index % 2} 0.5 0.5 0.4 0.4\n", encoding="utf-8")
    return processed


def _snapshot(root: Path) -> dict[str, str]:
    return {
        str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def test_export_reads_only_train_and_leaves_processed_untouched(tmp_path: Path) -> None:
    processed = _make_processed(tmp_path)
    before = _snapshot(processed)
    out = tmp_path / "augmented"
    policy = load_policy("proposed")

    summary = export_preview(policy, processed, out, per_class=None, n_variants=2, seed=1)

    assert _snapshot(processed) == before
    assert summary.source_images == 4 and summary.files_written == 8
    written = sorted(p.name for p in (out / "images" / "train").glob("*.jpg"))
    assert len(written) == 8 and all(name.startswith("tr_") for name in written)
    assert not list((out / "images").glob("val")) and not list((out / "images").glob("test"))
    for label in (out / "labels" / "train").glob("*.txt"):
        for line in label.read_text(encoding="utf-8").splitlines():
            class_id, *coords = line.split()
            assert int(class_id) in (0, 1) and len(coords) == 4
            assert all(0.0 <= float(value) <= 1.0 for value in coords)
    assert (out / "manifest.json").is_file()
    assert {p.name for p in summary.sheets} == {"paper.jpg", "glass.jpg"}


def test_export_samples_a_stratified_subset_of_train(tmp_path: Path) -> None:
    processed = _make_processed(tmp_path, per_split=6)
    out = tmp_path / "augmented"

    summary = export_preview(load_policy("proposed"), processed, out, per_class=2, n_variants=2)

    assert summary.source_images == 4 and summary.files_written == 8
    classes = [
        int(label.read_text(encoding="utf-8").split()[0])
        for label in (out / "labels" / "train").glob("*.txt")
    ]
    assert sorted(classes) == [0] * 4 + [1] * 4


def test_export_refuses_paths_that_overlap_protected_data(tmp_path: Path) -> None:
    processed = _make_processed(tmp_path)
    raw = tmp_path / "raw"
    (raw / "export").mkdir(parents=True)
    policy = load_policy("proposed")
    for bad in (processed, processed / "nested", tmp_path):
        with pytest.raises(ValueError, match="overlaps"):
            export_preview(policy, processed, bad, per_class=1, n_variants=1)
    with pytest.raises(ValueError, match="overlaps"):
        export_preview(
            policy, processed, raw / "export", per_class=1, n_variants=1, protected=[raw]
        )


def test_export_replaces_previous_output_and_keeps_gitkeep(tmp_path: Path) -> None:
    processed = _make_processed(tmp_path)
    out = tmp_path / "augmented"
    (out / "images" / "train").mkdir(parents=True)
    (out / "images" / "train" / "stale.jpg").write_bytes(b"old")
    (out / ".gitkeep").touch()
    policy = load_policy("proposed")

    export_preview(policy, processed, out, per_class=2, n_variants=1)
    first = sorted(p.name for p in (out / "images" / "train").glob("*.jpg"))
    export_preview(policy, processed, out, per_class=2, n_variants=1)
    second = sorted(p.name for p in (out / "images" / "train").glob("*.jpg"))

    assert "stale.jpg" not in first and first == second
    assert (out / ".gitkeep").is_file()


def test_export_can_take_every_train_image(tmp_path: Path) -> None:
    processed = _make_processed(tmp_path, per_split=5)
    summary = export_preview(
        load_policy("no_augmentation"),
        processed,
        tmp_path / "augmented",
        per_class=None,
        n_variants=1,
    )
    assert summary.source_images == 5 and summary.files_written == 5
