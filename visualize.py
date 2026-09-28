#!/usr/bin/env python
"""visualize.py — Visualize mammography annotations or model predictions.

Usage 1: Dataset Inspection (Raw 16-bit vs Ground Truth annotations):
    python visualize.py \
        --val-json    /path/to/mass_val.json \
        --images-dir  /path/to/images \
        --num-images  3 \
        --save-dir    ./viz_dataset

Usage 2: Inference Comparison (Ground Truth vs Model Predictions side-by-side):
    python visualize.py \
        --weights     output/model_best.pth \
        --config-file configs/mammo_dinov2_dino.py \
        --val-json    /path/to/mass_val.json \
        --images-dir  /path/to/images \
        --conf-thresh 0.30 \
        --num-images  4 \
        --save-dir    ./viz_predictions
"""

from __future__ import annotations

import argparse
import logging
import os
import sys

_local_detrex = os.path.join(os.path.dirname(os.path.abspath(__file__)), "detrex")
_default_detrex = os.environ.get(
    "DETREX_ROOT",
    _local_detrex if os.path.isdir(_local_detrex) else "/content/detrex"
)
sys.path.insert(0, _default_detrex)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from rfdetr.data.registration import register_mammo_dataset
from rfdetr.utils.visualize import visualize_crop, visualize_dataset, visualize_predictions

logger = logging.getLogger("mammo_visualize")


def main():
    parser = argparse.ArgumentParser(description="Visualize mammography annotations or model predictions")
    parser.add_argument("--val-json",     required=True, help="Path to COCO validation JSON")
    parser.add_argument("--images-dir",   required=True, help="Root directory for images")
    parser.add_argument("--train-json",   default=None,  help="Path to COCO training JSON (optional)")
    parser.add_argument("--weights",      default=None,  help="Checkpoint .pth path for model prediction mode")
    parser.add_argument("--config-file",  default="configs/mammo_dinov2_dino.py", help="LazyConfig .py path")
    parser.add_argument("--conf-thresh",  type=float, default=0.30, help="Confidence threshold for predictions (default: 0.30)")
    parser.add_argument("--split",        choices=["train", "val"], default="val", help="Dataset split to visualize (default: val)")
    parser.add_argument("--num-images",   type=int, default=3, help="Number of images to visualize (default: 3)")
    parser.add_argument("--low-pct",      type=float, default=1.0)
    parser.add_argument("--high-pct",     type=float, default=99.0)
    parser.add_argument("--test-size",    type=int, default=812, help="Shortest edge size for inference resize (default: 812)")
    parser.add_argument("--max-size",     type=int, default=1624, help="Maximum edge size for inference resize (default: 1624)")
    parser.add_argument("--save-dir",     default=None, help="Save figures here (optional, if omitted or if --show is passed, displays inline)")
    parser.add_argument("--show",         action="store_true", default=False, help="Force inline display (useful in Jupyter/Colab notebooks)")
    parser.add_argument("--crop",         action="store_true", default=False, help="Visualize LesionAwareCrop augmentation (518x518 crop with guaranteed lesion containment)")
    parser.add_argument("--crop-size",    type=int, nargs=2, default=[518, 518], help="Crop window height and width (default: 518 518)")
    parser.add_argument("--seed",         type=int, default=None, help="Random seed for reproducibility (default: None for fresh random selection)")
    parser.add_argument("--detrex-root",  default=_default_detrex)
    args = parser.parse_args()

    if args.detrex_root not in sys.path:
        sys.path.insert(0, args.detrex_root)

    train_json = args.train_json if args.train_json is not None else args.val_json

    thing_classes = register_mammo_dataset(
        train_json=train_json,
        val_json=args.val_json,
        images_dir=args.images_dir,
    )
    dataset_name = f"mammo_{args.split}"

    if args.weights:
        # ── Mode 2: Inference Comparison (GT vs Predictions) ─────────────────
        import torch
        from detectron2.checkpoint import DetectionCheckpointer
        from detectron2.config import LazyConfig, instantiate

        num_classes = len(thing_classes)
        cfg = LazyConfig.load(args.config_file)
        cfg.model.num_classes = num_classes
        if hasattr(cfg.model, "criterion") and hasattr(cfg.model.criterion, "num_classes"):
            cfg.model.criterion.num_classes = num_classes

        device = "cuda" if torch.cuda.is_available() else "cpu"
        cfg.model.device = device
        model = instantiate(cfg.model)
        model.to(device)
        model.eval()

        DetectionCheckpointer(model).load(args.weights)
        print(f"Loaded trained model weights from: {args.weights}")

        visualize_predictions(
            model=model,
            dataset_name=dataset_name,
            num_images=args.num_images,
            conf_thresh=args.conf_thresh,
            low_pct=args.low_pct,
            high_pct=args.high_pct,
            save_dir=args.save_dir,
            seed=args.seed,
            images_fallback_dir=args.images_dir,
            test_size=args.test_size,
            max_size=args.max_size,
            show=args.show,
        )
    elif args.crop:
        # ── Mode 3: Crop Augmentation Validation (Original vs LesionAwareCrop) ──
        visualize_crop(
            dataset_name=dataset_name,
            num_images=args.num_images,
            crop_size=tuple(args.crop_size),
            low_pct=args.low_pct,
            high_pct=args.high_pct,
            save_dir=args.save_dir,
            seed=args.seed,
            images_fallback_dir=args.images_dir,
            show=args.show,
        )
    else:
        # ── Mode 1: Dataset Inspection (Raw 16-bit vs GT annotations) ─────────
        visualize_dataset(
            dataset_name=dataset_name,
            num_images=args.num_images,
            low_pct=args.low_pct,
            high_pct=args.high_pct,
            save_dir=args.save_dir,
            seed=args.seed,
            images_fallback_dir=args.images_dir,
            show=args.show,
        )


if __name__ == "__main__":
    main()
