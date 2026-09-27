"""rfdetr.data.registration

Registers mammography COCO-format datasets into the detectron2 DatasetCatalog.

Classes are automatically extracted from the JSON 'categories' field — no
need to hardcode them. num_classes is then available via:
    MetadataCatalog.get("mammo_train").num_classes
    MetadataCatalog.get("mammo_train").thing_classes  # list of class names
"""

from __future__ import annotations

import json
import logging
from typing import List

try:
    from detectron2.data import DatasetCatalog, MetadataCatalog
    from detectron2.data.datasets import load_coco_json
except ImportError:
    DatasetCatalog = None
    MetadataCatalog = None
    load_coco_json = None

logger = logging.getLogger(__name__)

TRAIN_DATASET_NAME = "mammo_train"
VAL_DATASET_NAME = "mammo_val"


def get_classes_from_json(json_path: str) -> List[str]:
    """Extract ordered class names from a COCO annotation JSON.

    Reads the 'categories' field and returns class names sorted by category id.

    Args:
        json_path: Path to COCO-format annotation JSON.

    Returns:
        List of class name strings ordered by category id.
        Example: ["mass"] or ["mass", "calcification", "distortion"]
    """
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    categories = data.get("categories", [])
    if not categories:
        raise ValueError(f"No 'categories' found in {json_path}")

    # Sort by id to ensure consistent ordering
    categories_sorted = sorted(categories, key=lambda c: c["id"])
    thing_classes = [c["name"] for c in categories_sorted]

    logger.info(f"Found {len(thing_classes)} class(es) in {json_path}: {thing_classes}")
    return thing_classes


def register_mammo_dataset(
    train_json: str,
    val_json: str,
    images_dir: str,
    train_name: str = TRAIN_DATASET_NAME,
    val_name: str = VAL_DATASET_NAME,
) -> List[str]:
    """Register mammography train and val splits into detectron2 DatasetCatalog.

    Classes are auto-detected from the COCO JSON 'categories' field.

    Args:
        train_json: Path to COCO-format training annotation JSON.
        val_json: Path to COCO-format validation annotation JSON.
        images_dir: Root directory containing all images.
        train_name: Name for the training dataset (default: "mammo_train").
        val_name: Name for the validation dataset (default: "mammo_val").

    Returns:
        List of class names (thing_classes), e.g. ["mass"] or ["mass", "calcification"].
    """
    # Extract classes from train JSON (authoritative source)
    thing_classes = get_classes_from_json(train_json)

    for name, json_path in [(train_name, train_json), (val_name, val_json)]:
        # Safely unregister if previously registered (allows switching dataset JSON in same session)
        if DatasetCatalog is not None:
            if hasattr(DatasetCatalog, "_REGISTERED"):
                DatasetCatalog._REGISTERED.pop(name, None)
        if MetadataCatalog is not None:
            if hasattr(MetadataCatalog, "_REGISTERED"):
                MetadataCatalog._REGISTERED.pop(name, None)

        if DatasetCatalog is not None and load_coco_json is not None:
            DatasetCatalog.register(
                name,
                lambda j=json_path, d=images_dir, n=name: load_coco_json(j, d, n),
            )
            MetadataCatalog.get(name).set(
                thing_classes=thing_classes,
                num_classes=len(thing_classes),
                json_file=json_path,
                image_root=images_dir,
                evaluator_type="coco",
            )
        logger.info(f"Registered '{name}' — {len(thing_classes)} class(es): {thing_classes}")

    return thing_classes

