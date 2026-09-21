from pathlib import Path

import numpy as np
from PIL import Image

from vpc2.data import io


def test_list_images_non_recursive(tmp_path: Path) -> None:
    nested = tmp_path / "nested"
    nested.mkdir()
    Image.new("RGB", (8, 8), "red").save(tmp_path / "a.jpg")
    Image.new("RGB", (8, 8), "blue").save(nested / "b.jpg")
    (tmp_path / "notes.txt").write_text("ignore", encoding="utf-8")

    found = io.list_images(tmp_path, recursive=False)
    assert [path.name for path in found] == ["a.jpg"]

    recursive = io.list_images(tmp_path, recursive=True)
    assert {path.name for path in recursive} == {"a.jpg", "b.jpg"}


def test_load_image_converts_rgba_and_bgr(tmp_path: Path) -> None:
    path = tmp_path / "alpha.png"
    Image.new("RGBA", (4, 4), (255, 0, 0, 128)).save(path)

    rgb = io.load_image(path, mode="RGB")
    assert rgb.shape == (4, 4, 3)
    assert rgb.dtype == np.uint8
    np.testing.assert_array_equal(rgb[0, 0], [255, 0, 0])

    bgr = io.load_image(path, mode="BGR")
    np.testing.assert_array_equal(bgr[0, 0], [0, 0, 255])


def test_class_names_from_yaml_list_and_dict() -> None:
    assert io.class_names_from_yaml({"names": ["a", "b"]}) == ["a", "b"]
    assert io.class_names_from_yaml({"names": {1: "b", 0: "a"}}) == ["a", "b"]


def test_discover_raw_dataset_prefers_canonical_name(tmp_path: Path) -> None:
    canonical = tmp_path / "garbage-classification-3"
    canonical.mkdir()
    (canonical / "data.yaml").write_text("nc: 1\nnames: ['x']\n", encoding="utf-8")
    other = tmp_path / "other"
    other.mkdir()
    (other / "data.yaml").write_text("nc: 1\nnames: ['y']\n", encoding="utf-8")

    found = io.discover_raw_dataset(tmp_path)
    assert found == canonical.resolve()
