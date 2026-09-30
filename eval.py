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

_local_detrex = os.path.join(os.path.dirname(os.path.abspath(__file__)), "detrex")
_default_detrex = os.environ.get(
    "DETREX_ROOT",
    _local_detrex if os.path.isdir(_local_detrex) else "/content/detrex"
)
sys.path.insert(0, _default_detrex)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

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
    parser.add_argument("--num-workers", type=int, default=0, help="DataLoader workers (default: 0 for zero shared-memory overhead)")
    parser.add_argument("--visualize",   dest="visualize", action="store_true", default=True,
                        help="Generate side-by-side comparison figure: Column 1 = Ground Truth bboxes, Column 2 = Model Prediction bboxes (default: True)")
    parser.add_argument("--no-visualize", "--no-viz", dest="visualize", action="store_false",
                        help="Disable side-by-side comparison figure generation")
    parser.add_argument("--visualize-only", action="store_true", default=False,
                        help="Only generate the side-by-side Ground Truth vs Predictions figure without running the full test set COCO evaluation")
    parser.add_argument("--num-viz",     type=int, default=4,
                        help="Number of images in the side-by-side comparison figure (default: 4)")
    parser.add_argument("--conf-thresh", type=float, default=0.30,
                        help="Confidence threshold for predictions in comparison figure (default: 0.30)")
    parser.add_argument("--show",        action="store_true", default=False,
                        help="Display visualization inline in interactive notebooks via plt.show()")
    parser.add_argument("--detrex-root", default=_default_detrex)
    parser.add_argument("--opts", dest="named_opts", nargs="+", action="extend", default=[],
                        help="Modify config options using key=value")
    parser.add_argument("opts", default=None, nargs=argparse.REMAINDER,
                        help="Modify config options using key=value")
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
    all_opts = (args.opts or []) + (getattr(args, "named_opts", []) or [])
    cfg = LazyConfig.apply_overrides(cfg, all_opts)

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

    if not args.visualize_only:
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

    if args.visualize or args.visualize_only or args.show:
        from rfdetr.utils.visualize import visualize_predictions

        print("\n" + "=" * 70)
        print(f"📊 Génération de la comparaison Côtes-à-Côtes ({args.num_viz} images) :")
        print("   - Colonne 1 : Vraies Bounding Boxes (Ground Truth)")
        print("   - Colonne 2 : Prédictions Bounding Boxes Modèle RF-DETR")
        print("=" * 70)
        saved_fig = visualize_predictions(
            model=model,
            dataset_name="mammo_val",
            num_images=args.num_viz,
            conf_thresh=args.conf_thresh,
            save_dir=args.output_dir,
            images_fallback_dir=args.images_dir,
            test_size=args.test_size,
            max_size=args.max_size,
            show=args.show,
        )
        if saved_fig:
            print(f"✅ Figure de comparaison enregistrée : {saved_fig}")
            print("Pour l'afficher directement dans Google Colab, exécutez dans une cellule :")
            print("    from IPython.display import Image, display")
            print(f"    display(Image('{saved_fig}'))")
            print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
