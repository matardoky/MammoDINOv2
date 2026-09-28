#!/usr/bin/env python
"""scripts/create_overfit_subset.py

Creates a small, balanced COCO subset (20–50 images) from a full mammography dataset.
All selected images are guaranteed to contain active lesions across each class
(Mass, Asymmetry, ArchDistortion) for rapid architecture validation and overfit tests.

Usage:
    python scripts/create_overfit_subset.py \
        --input-json  /path/to/full_coco_3class_train.json \
        --output-json /path/to/overfit_subset_30.json \
        --num-images  30 \
        --seed        42
"""

from __future__ import annotations

import argparse
import json
import logging
import random
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Set

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s: %(message)s")
logger = logging.getLogger("create_overfit_subset")


def create_overfit_subset(
    input_json: str,
    output_json: str,
    num_images: int = 30,
    seed: int = 42,
) -> Dict[str, Any]:
    """Sample a balanced subset of images containing active annotations.

    Args:
        input_json: Path to the input COCO JSON.
        output_json: Path where the subset COCO JSON will be saved.
        num_images: Total number of images in the subset (default: 30).
        seed: Random seed for reproducibility (default: 42).

    Returns:
        The generated subset dictionary in COCO format.
    """
    random.seed(seed)
    input_path = Path(input_json)
    if not input_path.is_file():
        raise FileNotFoundError(f"Input JSON not found: {input_json}")

    logger.info(f"Loading COCO dataset from: {input_json}")
    with open(input_path, "r", encoding="utf-8") as f:
        coco_data = json.load(f)

    categories = coco_data.get("categories", [])
    images = coco_data.get("images", [])
    annotations = coco_data.get("annotations", [])

    logger.info(
        f"Input dataset: {len(images)} images, {len(annotations)} annotations, {len(categories)} categories"
    )

    # 1. Map images to annotations and categories
    img_id_to_img = {img["id"]: img for img in images}
    img_id_to_annos: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
    cat_id_to_name = {c["id"]: c["name"] for c in categories}
    cat_to_img_ids: Dict[int, List[int]] = defaultdict(list)

    for ann in annotations:
        if ann.get("iscrowd", 0) == 0:
            iid = ann["image_id"]
            cid = ann["category_id"]
            img_id_to_annos[iid].append(ann)
            cat_to_img_ids[cid].append(iid)

    # Filter only images with at least one valid annotation
    annotated_img_ids = [iid for iid, ann_list in img_id_to_annos.items() if len(ann_list) > 0]
    if len(annotated_img_ids) < num_images:
        logger.warning(
            f"Requested {num_images} images, but only {len(annotated_img_ids)} annotated images exist. "
            f"Using all {len(annotated_img_ids)} annotated images."
        )
        num_images = len(annotated_img_ids)

    # 2. Balanced sampling across categories
    selected_img_ids: Set[int] = set()
    n_cats = len(categories)
    per_cat = max(1, num_images // max(1, n_cats))

    for cat in categories:
        cid = cat["id"]
        cname = cat["name"]
        available = list(set(cat_to_img_ids.get(cid, [])) - selected_img_ids)
        random.shuffle(available)
        sampled = available[:per_cat]
        selected_img_ids.update(sampled)
        logger.info(f"  Category '{cname}' (id={cid}): sampled {len(sampled)} images")

    # If we need more images to reach target num_images, sample from remaining annotated images
    if len(selected_img_ids) < num_images:
        remaining = list(set(annotated_img_ids) - selected_img_ids)
        random.shuffle(remaining)
        needed = num_images - len(selected_img_ids)
        selected_img_ids.update(remaining[:needed])

    # 3. Assemble subset COCO structure
    subset_images = [img_id_to_img[iid] for iid in sorted(selected_img_ids)]
    subset_annotations = []
    ann_cat_counts: Dict[str, int] = defaultdict(int)

    for iid in sorted(selected_img_ids):
        for ann in img_id_to_annos[iid]:
            subset_annotations.append(ann)
            ann_cat_counts[cat_id_to_name.get(ann["category_id"], "unknown")] += 1

    subset_data = {
        "images": subset_images,
        "annotations": subset_annotations,
        "categories": categories,
    }

    # 4. Save to disk
    out_path = Path(output_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(subset_data, f, indent=2)

    logger.info("=" * 60)
    logger.info(f"Overfit subset successfully saved to: {output_json}")
    logger.info(f"Total images:      {len(subset_images)}")
    logger.info(f"Total annotations: {len(subset_annotations)}")
    logger.info("Annotation breakdown by class:")
    for cname, count in sorted(ann_cat_counts.items()):
        logger.info(f"  - {cname:20s}: {count} instance(s)")
    logger.info("=" * 60)

    return subset_data


def main():
    parser = argparse.ArgumentParser(description="Create overfit COCO subset for architecture testing")
    parser.add_argument("--input-json",  required=True, help="Path to input full COCO JSON")
    parser.add_argument("--output-json", default="overfit_subset_30.json", help="Path to output subset JSON")
    parser.add_argument("--num-images",  type=int, default=30, help="Number of images to sample (default: 30)")
    parser.add_argument("--seed",        type=int, default=42, help="Random seed (default: 42)")
    args = parser.parse_args()

    create_overfit_subset(
        input_json=args.input_json,
        output_json=args.output_json,
        num_images=args.num_images,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
