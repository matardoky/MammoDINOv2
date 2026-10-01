#!/usr/bin/env python
"""overfit.py — Fast Overfit Verification Module for RF-DETR.

Verifies end-to-end model learning capacity on a small micro-batch of 20 to 50 images:
1. Automatically extracts a balanced subset of annotated images from the full dataset.
2. Trains the model with fast updates (grad_accum_steps=1, batch_size=2).
3. Evaluates periodically on the SAME subset to observe loss collapse and AP50 convergence.
4. Optionally renders side-by-side visual predictions proving the model locks onto the lesions.

Usage:
    python overfit.py \
        --train-json /content/drive/MyDrive/EMBED_Dataset/curated/full_dataset/coco/full_coco_3class_train.json \
        --images-dir /content/mammo_data/images \
        --dinov2-weights /content/drive/MyDrive/EMBED_Dataset/checkpoints/dinov2_latest_checkpoint.pth \
        --output-dir ./output_overfit \
        --num-images 30 \
        --max-iter 400 \
        --eval-period 50
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import warnings
from pathlib import Path

# Suppress non-critical deprecation warnings
warnings.filterwarnings("ignore", category=FutureWarning, module=r".*timm\.models\.layers.*")
warnings.filterwarnings("ignore", category=FutureWarning, module=r".*detectron2\.layers\.dcn_v3.*")
warnings.filterwarnings("ignore", category=FutureWarning, message=r".*torch\.cuda\.amp\.custom_.*")
warnings.filterwarnings("ignore", category=UserWarning, message=r".*pkg_resources is deprecated.*")
warnings.filterwarnings("ignore", category=UserWarning, message=r".*lr_scheduler\.step\(\).*before.*optimizer\.step\(\).*")

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# Setup Detrex import path
_local_detrex = os.path.join(os.path.dirname(os.path.abspath(__file__)), "detrex")
_default_detrex = os.environ.get(
    "DETREX_ROOT",
    _local_detrex if os.path.isdir(_local_detrex) else "/content/detrex"
)
sys.path.insert(0, _default_detrex)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "scripts"))

import torch
import detectron2.data.transforms as T
from detectron2.config import LazyCall as L, LazyConfig
from detectron2.engine import default_argument_parser, default_setup, launch
from detectron2.solver import WarmupParamScheduler
from fvcore.common.param_scheduler import MultiStepParamScheduler

from create_overfit_subset import create_overfit_subset
from rfdetr.data.registration import register_mammo_dataset
from train import do_train

logger = logging.getLogger("mammo_overfit")


def build_arg_parser():
    parser = default_argument_parser()
    parser.add_argument("--train-json",      required=True, help="Path to source full COCO train JSON")
    parser.add_argument("--images-dir",      required=True, help="Root directory for images")
    parser.add_argument("--dinov2-weights",  default=None,  help="Path to pretrained DINOv2 weights (.pth)")
    parser.add_argument("--output-dir",      default="./output_overfit", help="Output directory for checkpoints and logs")
    parser.set_defaults(config_file="configs/mammo_dinov2_dino.py")
    parser.add_argument("--num-images",      type=int, default=20, help="Number of images in overfit subset (default: 20)")
    parser.add_argument("--max-iter",        type=int, default=1000, help="Total training iterations (default: 1000)")
    parser.add_argument("--eval-period",     type=int, default=50, help="Evaluation and checkpoint period (default: 50)")
    parser.add_argument("--log-period",      type=int, default=10, help="Logging period (default: 10)")
    parser.add_argument("--batch-size",      type=int, default=2, help="Physical batch size (default: 2)")
    parser.add_argument("--lr",              type=float, default=1e-4, help="Base learning rate for heads/transformer (default: 1e-4)")
    parser.add_argument("--backbone-lr",     type=float, default=1.19e-4, help="Top ViT block learning rate (default: 1.19e-4)")
    parser.add_argument("--seed",            type=int, default=42, help="Random seed for subset sampling (default: 42)")
    parser.add_argument("--visualize-after", action="store_true", default=True, help="Produce visual GT vs prediction comparison at end")
    parser.add_argument("--detrex-root",     default=_default_detrex, help="Path to detrex clone")
    parser.add_argument("--freeze-blocks",   type=int, default=0, help="Number of DINOv2 blocks to freeze (default: 0 = all layers unfrozen)")
    parser.add_argument("--num-queries",     type=int, default=100, help="Number of object queries in DINO (default: 100)")
    parser.add_argument("--dn-number",       type=int, default=10, help="Number of denoising query groups (default: 10)")
    parser.add_argument("--clip-grad-norm",  type=float, default=0.1, help="Maximum gradient norm for clipping (default: 0.1)")
    parser.add_argument("--opts",            dest="named_opts", nargs="+", action="extend", default=[],
                        help="Optional config overrides (e.g. --opts train.max_iter=500)")
    return parser


def main(args):
    if args.detrex_root not in sys.path:
        sys.path.insert(0, args.detrex_root)

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Step 1: Create or reuse balanced overfit micro-subset ────────────────
    subset_json = output_dir / f"overfit_subset_{args.num_images}.json"
    if not subset_json.is_file():
        logger.info(f"Generating balanced overfit subset ({args.num_images} images) from {args.train_json}...")
        create_overfit_subset(
            input_json=args.train_json,
            output_json=str(subset_json),
            num_images=args.num_images,
            seed=args.seed,
        )
    else:
        logger.info(f"Reusing existing overfit subset: {subset_json}")

    # ── Step 2: Register subset as BOTH train and val dataset ─────────────────
    thing_classes = register_mammo_dataset(
        train_json=str(subset_json),
        val_json=str(subset_json),
        images_dir=args.images_dir,
    )
    num_classes = len(thing_classes)
    logger.info(f"Overfit dataset registered with {num_classes} classes: {thing_classes}")

    # ── Step 3: Configure fast overfit schedule ──────────────────────────────
    cfg = LazyConfig.load(args.config_file)

    # Apply any CLI overrides
    all_opts = (args.opts or []) + (getattr(args, "named_opts", []) or [])
    cfg = LazyConfig.apply_overrides(cfg, all_opts)

    # Inject dataset classes
    cfg.model.num_classes = num_classes
    if hasattr(cfg.model, "criterion") and hasattr(cfg.model.criterion, "num_classes"):
        cfg.model.criterion.num_classes = num_classes

    # Inject output dir and weights
    cfg.train.output_dir = str(output_dir)
    if args.dinov2_weights:
        cfg.model.backbone.backbone.checkpoint_path = args.dinov2_weights

    # Inject freeze_blocks (default 2 = first 2 blocks frozen)
    if hasattr(args, "freeze_blocks") and args.freeze_blocks is not None:
        cfg.model.backbone.backbone.freeze_blocks = args.freeze_blocks
        logger.info(f"Backbone freeze_blocks set to: {args.freeze_blocks}")

    # Inject num_queries & dn_number
    if getattr(args, "num_queries", None) is not None:
        cfg.model.num_queries = args.num_queries
        cfg.model.select_box_nums_for_evaluation = args.num_queries
        logger.info(f"Model num_queries set to: {args.num_queries}")
    if getattr(args, "dn_number", None) is not None:
        cfg.model.dn_number = args.dn_number
        logger.info(f"Model dn_number set to: {args.dn_number}")

    # Inject clip_grad_norm
    if getattr(args, "clip_grad_norm", None) is not None:
        if not hasattr(cfg.train, "clip_grad") or cfg.train.clip_grad is None:
            cfg.train.clip_grad = dict(enabled=True, params=dict(norm_type=2))
        cfg.train.clip_grad.enabled = True
        cfg.train.clip_grad.params.max_norm = args.clip_grad_norm
        logger.info(f"Gradient clipping max_norm set to: {args.clip_grad_norm}")

    # Inject learning rate into optimizer
    if hasattr(cfg, "optimizer"):
        cfg.optimizer.lr = args.lr
        if hasattr(cfg.optimizer, "params"):
            if hasattr(cfg.optimizer.params, "base_lr"):
                cfg.optimizer.params.base_lr = args.lr
            if hasattr(cfg.optimizer.params, "backbone_lr"):
                cfg.optimizer.params.backbone_lr = args.backbone_lr

    # Inject image directory into mappers — keeping exact training transforms and resize intact
    if hasattr(cfg, "dataloader"):
        if hasattr(cfg.dataloader, "train"):
            # Clean infinite TrainingSampler without RepeatFactor distortion for small overfit subset
            cfg.dataloader.train.sampler = None
            if hasattr(cfg.dataloader.train, "mapper"):
                cfg.dataloader.train.mapper.images_fallback_dir = args.images_dir
                cfg.dataloader.train.mapper.crop_prob = 0.0  # Full uncropped images for clean validation
                cfg.dataloader.train.total_batch_size = args.batch_size
                # Deterministic overfit: use the exact same fixed resize as test loader (no random flip/jitter)
                cfg.dataloader.train.mapper.augmentation = [
                    L(T.ResizeShortestEdge)(
                        short_edge_length=(812,),
                        max_size=1344,
                        sample_style="choice",
                    )
                ]
        if hasattr(cfg.dataloader, "test") and hasattr(cfg.dataloader.test, "mapper"):
            cfg.dataloader.test.mapper.images_fallback_dir = args.images_dir

    # Overfit runtime parameters: fast steps without gradient accumulation delay
    cfg.train.max_iter = args.max_iter
    cfg.train.eval_period = args.eval_period
    cfg.train.log_period = args.log_period
    cfg.train.grad_accum_steps = 1  # Immediate optimizer steps on every micro-batch
    cfg.train.checkpointer = dict(period=args.eval_period, max_to_keep=5)

    # Learning rate schedule: 20 steps warmup, decay at 70% and 90% matching reference
    warmup_steps = min(20, max(5, int(args.max_iter * 0.02)))
    milestone_1 = int(args.max_iter * 0.70)
    milestone_2 = int(args.max_iter * 0.90)
    cfg.lr_multiplier = L(WarmupParamScheduler)(
        scheduler=L(MultiStepParamScheduler)(
            values=[1.0, 0.1, 0.01],
            milestones=[milestone_1, milestone_2, args.max_iter],
        ),
        warmup_length=warmup_steps / args.max_iter,
        warmup_method="linear",
        warmup_factor=0.001,
    )

    default_setup(cfg, args)

    # ── Step 4: Execute Overfit Training & Evaluation ─────────────────────────
    logger.info("=" * 70)
    logger.info(f"STARTING OVERFIT TEST: {args.num_images} images | {args.max_iter} iters | eval every {args.eval_period} iters")
    logger.info("Watch the loss drop to near-zero and AP50 climb rapidly on the evaluation subset!")
    logger.info("=" * 70)

    do_train(args, cfg)

    # ── Step 5: Optional Visual Verification ─────────────────────────────────
    if args.visualize_after:
        best_model_path = output_dir / "model_best.pth"
        eval_weights = str(best_model_path if best_model_path.is_file() else output_dir / "model_final.pth")
        if os.path.isfile(eval_weights):
            logger.info("=" * 70)
            logger.info("Generating post-overfit visual predictions...")
            try:
                from detectron2.checkpoint import DetectionCheckpointer
                from detectron2.config import instantiate
                from rfdetr.utils.visualize import visualize_predictions

                device = "cuda" if torch.cuda.is_available() else "cpu"
                model = instantiate(cfg.model)
                model.to(device)
                model.eval()
                DetectionCheckpointer(model).load(eval_weights)

                visualize_predictions(
                    model=model,
                    dataset_name="mammo_val",
                    num_images=min(4, args.num_images),
                    conf_thresh=0.25,
                    save_dir=str(output_dir / "visualizations"),
                    images_fallback_dir=args.images_dir,
                    show=False,
                )
                logger.info(f"Visual predictions saved to: {output_dir / 'visualizations'}")
            except Exception as e:
                logger.warning(f"Visualization step failed: {e}")
        logger.info("=" * 70)
        logger.info("OVERFIT TEST COMPLETED SUCCESSFULLY!")


if __name__ == "__main__":
    args = build_arg_parser().parse_args()
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )
