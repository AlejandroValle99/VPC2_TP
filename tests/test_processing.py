from pathlib import Path

from PIL import Image

from vpc2.data.processing import (
    PairedSample,
    YoloBox,
    parse_yolo_label,
    run_pipeline,
    stratified_split,
    write_processed_dataset,
)


def _write_jpeg(path: Path, color: str = "red") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (32, 32), color).save(path, format="JPEG")


def test_parse_yolo_label_clips_dedupes_and_drops(tmp_path: Path) -> None:
    label = tmp_path / "sample.txt"
    label.write_text(
        (
            "0 0.5 0.5 0.2 0.2\n"
            "0 0.5 0.5 0.2 0.2\n"
            "0 0.95 0.5 0.2 0.2\n"
            "9 0.5 0.5 0.1 0.1\n"
            "not a box\n"
            "1 0.25 0.25 0.1 0.1\n"
        ),
        encoding="utf-8",
    )

    boxes, counters, issues = parse_yolo_label(label, num_classes=2, clip_out_of_bounds=True)

    assert [box.class_id for box in boxes] == [0, 0, 1]
    assert counters["duplicates_removed"] == 1
    assert counters["invalid_class"] == 1
    assert counters["malformed_lines"] == 1
    assert counters["boxes_clipped"] == 1
    assert any("clipped" in issue for issue in issues)
    assert boxes[1].x_center <= 1.0
    assert boxes[1].width <= 1.0


def test_parse_empty_label(tmp_path: Path) -> None:
    label = tmp_path / "empty.txt"
    label.write_text("", encoding="utf-8")
    boxes, _counters, issues = parse_yolo_label(label, num_classes=2)
    assert boxes == []
    assert any("empty_label" in issue for issue in issues)


def test_parse_yolo_label_drops_out_of_bounds_when_not_clipping(tmp_path: Path) -> None:
    label = tmp_path / "sample.txt"
    label.write_text("0 0.95 0.5 0.2 0.2\n0 0.5 0.5 0.2 0.2\n", encoding="utf-8")

    boxes, counters, issues = parse_yolo_label(label, num_classes=2, clip_out_of_bounds=False)

    assert [box.class_id for box in boxes] == [0]
    assert counters["boxes_dropped"] == 1
    assert any("out of bounds" in issue for issue in issues)


def test_stratified_split_is_reproducible() -> None:
    samples = [
        PairedSample(
            stem=f"s{index}",
            image_path=Path(f"{index}.jpg"),
            label_path=Path(f"{index}.txt"),
            boxes=[YoloBox(class_id=index % 2, x_center=0.5, y_center=0.5, width=0.2, height=0.2)],
            converted=False,
        )
        for index in range(20)
    ]
    first = stratified_split(samples, seed=42)
    second = stratified_split(samples, seed=42)
    assert [sample.stem for sample in first["train"]] == [sample.stem for sample in second["train"]]
    assert len(first["train"]) + len(first["val"]) + len(first["test"]) == 20
    train_stems = {sample.stem for sample in first["train"]}
    val_stems = {sample.stem for sample in first["val"]}
    test_stems = {sample.stem for sample in first["test"]}
    assert train_stems.isdisjoint(val_stems)
    assert train_stems.isdisjoint(test_stems)
    assert val_stems.isdisjoint(test_stems)


def _make_raw_dataset(root: Path) -> Path:
    dataset = root / "garbage-classification-3"
    (dataset / "train" / "images").mkdir(parents=True)
    (dataset / "train" / "labels").mkdir(parents=True)
    (dataset / "data.yaml").write_text(
        "nc: 2\nnames: ['paper', 'plastic']\n",
        encoding="utf-8",
    )

    for index in range(8):
        class_id = index % 2
        stem = f"item_{index}"
        _write_jpeg(dataset / "train" / "images" / f"{stem}.jpg")
        (dataset / "train" / "labels" / f"{stem}.txt").write_text(
            f"{class_id} 0.5 0.5 0.4 0.4\n",
            encoding="utf-8",
        )

    # Orphan image, orphan label, corrupt image, RGBA image, duplicate box, OOB box.
    _write_jpeg(dataset / "train" / "images" / "orphan_image.jpg")
    (dataset / "train" / "labels" / "orphan_label.txt").write_text(
        "0 0.5 0.5 0.2 0.2\n", encoding="utf-8"
    )
    (dataset / "train" / "images" / "broken.jpg").write_bytes(b"not-an-image")
    Image.new("RGBA", (16, 16), (0, 255, 0, 90)).save(
        dataset / "train" / "images" / "rgba_item.png"
    )
    (dataset / "train" / "labels" / "rgba_item.txt").write_text(
        "1 0.5 0.5 0.3 0.3\n1 0.5 0.5 0.3 0.3\n1 0.95 0.5 0.2 0.2\n",
        encoding="utf-8",
    )
    (dataset / "train" / "labels" / "empty_pair.txt").write_text("", encoding="utf-8")
    _write_jpeg(dataset / "train" / "images" / "empty_pair.jpg")
    return dataset


def test_write_processed_dataset_preserves_gitkeep(tmp_path: Path) -> None:
    processed = tmp_path / "processed"
    processed.mkdir()
    (processed / ".gitkeep").touch()
    (processed / "stale_file.txt").write_text("old", encoding="utf-8")

    image_path = tmp_path / "s0.jpg"
    _write_jpeg(image_path)
    sample = PairedSample(
        stem="s0",
        image_path=image_path,
        label_path=tmp_path / "s0.txt",
        boxes=[YoloBox(class_id=0, x_center=0.5, y_center=0.5, width=0.2, height=0.2)],
        converted=False,
    )

    write_processed_dataset({"train": [sample]}, processed, class_names=["a"])

    assert (processed / ".gitkeep").is_file()
    assert not (processed / "stale_file.txt").exists()


def test_run_pipeline_end_to_end(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    _make_raw_dataset(raw)
    interim = tmp_path / "interim"
    processed = tmp_path / "processed"

    report = run_pipeline(
        raw_dir=raw,
        interim_dir=interim,
        processed_dir=processed,
        seed=0,
        ratios=(0.5, 0.25, 0.25),
    )

    assert report.orphan_images == 2  # orphan_image.jpg + corrupt broken.jpg, both label-less
    assert report.orphan_labels == 1
    assert report.images_corrupt == 1
    assert report.empty_labels == 1
    assert report.images_converted >= 1
    assert report.duplicates_removed == 1
    assert report.paired_kept == 9  # 8 clean RGB + 1 RGBA
    assert report.split_train + report.split_val + report.split_test == 9

    yaml_path = processed / "data.yaml"
    assert yaml_path.is_file()
    assert (processed / "images" / "train").is_dir()
    assert (processed / "labels" / "train").is_dir()
    assert (interim / "audit" / "report.json").is_file()
    assert (interim / "orphans" / "orphan_images.txt").read_text(encoding="utf-8").strip()
    assert (interim / "orphans" / "orphan_labels.txt").read_text(encoding="utf-8").strip()
