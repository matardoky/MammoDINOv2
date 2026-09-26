#!/usr/bin/env python
"""eval.py — Standalone COCO evaluation for mammography lesion detection.

Usage:
    python eval.py \\
        --config-file configs/mammo_dinov2_dino.py \\
        --weights output/model_final.pth \\
        --val-json /path/to/mass_val.json \\
        --images-dir /path/to/images \\
        --output-dir ./eval_output \\
        --detrex-root /content/detrex
"""

import argparse
import logging
import os
import sys

_default_detrex = os.environ.get("DETREX_ROOT", "/content/detrex")
sys.path.insert(0, _default_detrex)
sys.path.insert(0, os.path.dirname(__file__))

import detectron2.data.transforms as T
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import LazyConfig, instantiate
from detectron2.data import build_detection_test_loader
from detectron2.evaluation import COCOEvaluator, inference_on_dataset, print_csv_format

from rfdetr.data.mapper import Mammo16BitMapper
from rfdetr.data.registration import register_mammo_dataset

logger = logging.getLogger("mammo_eval")


def main():
    parser = argparse.ArgumentParser(description="Mammography DINO evaluation")
    parser.add_argument("--config-file", required=True, help="LazyConfig .py path")
    parser.add_argument("--weights",     required=True, help="Checkpoint .pth path")
    parser.add_argument("--val-json",    required=True, help="COCO val JSON path")
    parser.add_argument("--images-dir",  required=True, help="Images root directory")
    parser.add_argument("--output-dir",  default="./eval_output")
    parser.add_argument("--test-size",   type=int, default=812)
    parser.add_argument("--max-size",    type=int, default=1624)
    parser.add_argument("--num-workers", type=int, default=1)
    parser.add_argument("--detrex-root", default=_default_detrex)
    args = parser.parse_args()

    if args.detrex_root not in sys.path:
        sys.path.insert(0, args.detrex_root)

    os.makedirs(args.output_dir, exist_ok=True)

    thing_classes = register_mammo_dataset(
        train_json=args.val_json,  # val JSON as source of class names
        val_json=args.val_json,
        images_dir=args.images_dir,
    )
    num_classes = len(thing_classes)

    cfg = LazyConfig.load(args.config_file)

    # Auto-patch num_classes from dataset JSON
    cfg.model.num_classes = num_classes
    if hasattr(cfg.model, "criterion") and hasattr(cfg.model.criterion, "num_classes"):
        cfg.model.criterion.num_classes = num_classes
    logger.info(f"Auto-detected {num_classes} class(es): {thing_classes}")

    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg.model.device = device
    model = instantiate(cfg.model)
    model.to(device)
    model.eval()

    DetectionCheckpointer(model).load(args.weights)
    logger.info(f"Loaded weights from {args.weights}")

    from detectron2.data import DatasetCatalog

    val_dataset_dicts = DatasetCatalog.get("mammo_val")
    test_loader = build_detection_test_loader(
        dataset=val_dataset_dicts,
        mapper=Mammo16BitMapper(
            augmentation=[
                T.ResizeShortestEdge(
                    short_edge_length=(args.test_size,),
                    max_size=args.max_size,
                    sample_style="choice",
                )
            ],
            augmentation_with_crop=None,
            is_train=False,
            images_fallback_dir=args.images_dir,
        ),
        num_workers=args.num_workers,
    )

    evaluator = COCOEvaluator(
        dataset_name="mammo_val",
        output_dir=args.output_dir,
    )

    results = inference_on_dataset(model, test_loader, evaluator)
    print_csv_format(results)


if __name__ == "__main__":
    main()
