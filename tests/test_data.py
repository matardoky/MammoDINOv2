"""tests/test_data.py

Automated unit tests for data ingestion, 16-bit mammography normalization,
and COCO dataset registration utilities.
"""

from __future__ import annotations

import json
import os
import tempfile
import cv2
import numpy as np
import pytest
import torch

from rfdetr.data.mapper import Mammo16BitMapper, normalize_with_percentiles
from rfdetr.data.registration import get_classes_from_json
from rfdetr.utils.visualize import read_mammo_uint8


# ─── Fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture
def tmp_dir():
    with tempfile.TemporaryDirectory() as d:
        yield d


@pytest.fixture
def synthetic_uint16_image(tmp_dir):
    """Create a synthetic 16-bit grayscale image (512x512, uint16, values up to 60000)."""
    img_16 = np.linspace(1000, 60000, 512 * 512, dtype=np.uint16).reshape((512, 512))
    path = os.path.join(tmp_dir, "synthetic_16bit.png")
    cv2.imwrite(path, img_16)
    return path


@pytest.fixture
def synthetic_uint8_image(tmp_dir):
    """Create a synthetic 8-bit grayscale image."""
    img_8 = np.random.randint(0, 256, (256, 256), dtype=np.uint8)
    path = os.path.join(tmp_dir, "synthetic_8bit.png")
    cv2.imwrite(path, img_8)
    return path


@pytest.fixture
def synthetic_flat_image(tmp_dir):
    """Create a constant/flat image to test zero-range percentile protection."""
    img_flat = np.full((128, 128), 5000, dtype=np.uint16)
    path = os.path.join(tmp_dir, "synthetic_flat.png")
    cv2.imwrite(path, img_flat)
    return path


@pytest.fixture
def synthetic_coco_json(tmp_dir):
    """Create a valid COCO annotation JSON with 3 imbalanced classes."""
    data = {
        "images": [
            {"id": 1, "file_name": "img1.png", "height": 512, "width": 512},
            {"id": 2, "file_name": "img2.png", "height": 512, "width": 512},
        ],
        "categories": [
            {"id": 1, "name": "Asymmetry"},
            {"id": 2, "name": "Mass"},
            {"id": 3, "name": "ArchDistortion"},
        ],
        "annotations": [
            {"id": 1, "image_id": 1, "category_id": 1, "bbox": [50, 50, 100, 100], "area": 10000, "iscrowd": 0},
            {"id": 2, "image_id": 2, "category_id": 3, "bbox": [100, 100, 40, 40], "area": 1600, "iscrowd": 0},
        ],
    }
    path = os.path.join(tmp_dir, "annotations.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f)
    return path


# ─── 1. Category extraction from JSON ─────────────────────────────────────────

def test_get_classes_from_json_ordered(synthetic_coco_json):
    """Verify class names are correctly extracted and sorted by category id."""
    classes = get_classes_from_json(synthetic_coco_json)
    assert classes == ["Asymmetry", "Mass", "ArchDistortion"]
    assert len(classes) == 3


def test_get_classes_from_json_unordered(tmp_dir):
    """Verify classes with non-sequential or out-of-order IDs are correctly sorted."""
    data = {
        "categories": [
            {"id": 10, "name": "ArchDistortion"},
            {"id": 2,  "name": "Asymmetry"},
            {"id": 5,  "name": "Mass"},
        ]
    }
    path = os.path.join(tmp_dir, "unordered.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f)

    classes = get_classes_from_json(path)
    assert classes == ["Asymmetry", "Mass", "ArchDistortion"]


def test_get_classes_from_json_empty_categories(tmp_dir):
    """Verify ValueError is raised if categories is missing or empty."""
    path = os.path.join(tmp_dir, "empty_cat.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"images": []}, f)

    with pytest.raises(ValueError, match="No 'categories' found"):
        get_classes_from_json(path)


# ─── 2. Mammo16BitMapper Image Reading & Normalization ───────────────────────

def test_mammo_mapper_uint16_normalization(synthetic_uint16_image):
    """Verify 16-bit uint16 image is converted to float32 RGB in [0.0, 1.0] without quantization."""
    mapper = Mammo16BitMapper(augmentation=[], is_train=False)
    img_rgb = mapper._read_image(synthetic_uint16_image)

    assert isinstance(img_rgb, np.ndarray)
    assert img_rgb.dtype == np.float32
    assert img_rgb.shape == (512, 512, 3)
    # Check that grayscale is replicated identically across RGB channels
    assert np.array_equal(img_rgb[:, :, 0], img_rgb[:, :, 1])
    assert np.array_equal(img_rgb[:, :, 1], img_rgb[:, :, 2])
    # Check intensity range is within [0.0, 1.0]
    assert 0.0 <= img_rgb.min()
    assert img_rgb.max() <= 1.0
    assert img_rgb.max() > 0.5


def test_mammo_mapper_uint8_support(synthetic_uint8_image):
    """Verify standard 8-bit grayscale image is properly converted to 3-channel float32 RGB in [0.0, 1.0]."""
    mapper = Mammo16BitMapper(augmentation=[], is_train=False)
    img_rgb = mapper._read_image(synthetic_uint8_image)

    assert img_rgb.dtype == np.float32
    assert img_rgb.shape == (256, 256, 3)
    assert np.array_equal(img_rgb[:, :, 0], img_rgb[:, :, 1])
    assert 0.0 <= img_rgb.min() <= img_rgb.max() <= 1.0


def test_mammo_mapper_constant_image_protection(synthetic_flat_image):
    """Verify constant/flat image (hi <= lo) does not produce division-by-zero or NaNs."""
    mapper = Mammo16BitMapper(augmentation=[], is_train=False)
    img_rgb = mapper._read_image(synthetic_flat_image)

    assert img_rgb.dtype == np.float32
    assert not np.isnan(img_rgb).any()
    assert not np.isinf(img_rgb).any()
    assert img_rgb.shape == (128, 128, 3)


def test_normalize_with_percentiles_foreground_filter():
    """Verify normalize_with_percentiles filters out background air (pixels == 0)."""
    # 70% black background (0), 30% breast tissue (values 10000 to 50000)
    arr = np.zeros((300, 300), dtype=np.uint16)
    arr[100:250, 100:250] = np.linspace(10000, 50000, 150 * 150, dtype=np.uint16).reshape((150, 150))

    norm = normalize_with_percentiles(arr, low_pct=1.0, high_pct=99.0)
    assert norm.dtype == np.float32
    assert norm.shape == (300, 300)
    # Background remains 0.0
    assert norm[0, 0] == 0.0
    # Foreground pixels are normalized into [0.0, 1.0]
    fg_norm = norm[100:250, 100:250]
    assert fg_norm.min() >= 0.0
    assert fg_norm.max() <= 1.0
    assert fg_norm.max() > 0.9


# ─── 3. Path Resolution & Fallback ───────────────────────────────────────────

def test_resolve_path_direct_success(synthetic_uint16_image):
    """Path exists directly -> returned unchanged."""
    mapper = Mammo16BitMapper(augmentation=[], is_train=False)
    resolved = mapper._resolve_path(synthetic_uint16_image)
    assert resolved == synthetic_uint16_image


def test_resolve_path_fallback_success(tmp_dir, synthetic_uint16_image):
    """File moved/relative -> resolved in images_fallback_dir."""
    fake_path = "/non_existent_mount/data/" + os.path.basename(synthetic_uint16_image)
    mapper = Mammo16BitMapper(augmentation=[], is_train=False, images_fallback_dir=tmp_dir)
    resolved = mapper._resolve_path(fake_path)

    assert os.path.isfile(resolved)
    assert resolved == synthetic_uint16_image


def test_resolve_path_failure_raises(tmp_dir):
    """File not found anywhere -> raises FileNotFoundError."""
    mapper = Mammo16BitMapper(augmentation=[], is_train=False, images_fallback_dir=tmp_dir)
    with pytest.raises(FileNotFoundError, match="Image not found"):
        mapper._resolve_path("completely_missing.png")


# ─── 4. read_mammo_uint8 in visualize utility ─────────────────────────────────

def test_read_mammo_uint8_standalone(synthetic_uint16_image):
    """Verify standalone utility read_mammo_uint8 works identically."""
    img = read_mammo_uint8(synthetic_uint16_image, low_pct=1.0, high_pct=99.0)
    assert img.dtype == np.uint8
    assert img.shape == (512, 512, 3)
    assert np.array_equal(img[:, :, 0], img[:, :, 1])


def test_read_mammo_uint8_with_fallback(tmp_dir, synthetic_uint16_image):
    """Verify standalone utility handles fallback resolution."""
    fake_path = "non_existent_folder/" + os.path.basename(synthetic_uint16_image)
    img = read_mammo_uint8(fake_path, images_fallback_dir=tmp_dir)
    assert img.shape == (512, 512, 3)


# ─── 5. Class Imbalance RFS logic verification ───────────────────────────────

def test_repeat_factors_imbalance_ratio():
    """Verify repeat factors prioritize minority classes (e.g. ArchDistortion ~7%)."""
    # 100 images: 68 Asymmetry (id=1), 25 Mass (id=2), 7 ArchDistortion (id=3)
    dataset_dicts = (
        [{"annotations": [{"category_id": 1}]} for _ in range(68)]
        + [{"annotations": [{"category_id": 2}]} for _ in range(25)]
        + [{"annotations": [{"category_id": 3}]} for _ in range(7)]
    )

    repeat_thresh = 0.15

    # Compute frequency per category
    from collections import defaultdict
    freq = defaultdict(int)
    for d in dataset_dicts:
        for cat in set(a["category_id"] for a in d["annotations"]):
            freq[cat] += 1
    total = len(dataset_dicts)
    freq = {k: v / total for k, v in freq.items()}

    # Compute factors
    factors = []
    for d in dataset_dicts:
        cats = [a["category_id"] for a in d["annotations"]]
        r = max(max(1.0, (repeat_thresh / freq[c]) ** 0.5) for c in cats)
        factors.append(r)

    # Asymmetry (68% > 15% threshold) should NOT be repeated: factor == 1.0
    assert factors[0] == 1.0

    # Mass (25% > 15% threshold) should NOT be repeated: factor == 1.0
    assert factors[70] == 1.0

    # ArchDistortion (7% < 15% threshold) MUST be oversampled: factor > 1.4
    arch_dist_factor = factors[-1]
    expected_factor = (0.15 / 0.07) ** 0.5
    assert abs(arch_dist_factor - expected_factor) < 1e-4
    assert arch_dist_factor > 1.40
