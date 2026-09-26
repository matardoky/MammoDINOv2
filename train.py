#!/usr/bin/env python
"""train.py — Mammography DINO detection training launcher.

Thin wrapper around detrex's training infrastructure.
All training logic (SimpleTrainer, AMPTrainer, checkpointing, LR scheduling,
evaluation hooks, distributed launch) is handled by detrex / detectron2 directly.

Usage (single GPU):
    python train.py \\
        --config-file configs/mammo_dinov2_dino.py \\
        --train-json  /path/to/mass_train.json \\
        --val-json    /path/to/mass_val.json \\
        --images-dir  /path/to/images \\
        --dinov2-weights /path/to/dinov2_checkpoint.pth \\
        --output-dir  ./output \\
        --opts train.max_iter=7200 train.eval_period=600

Usage (Colab / single GPU, no dist):
    python train.py --config-file configs/mammo_dinov2_dino.py \\
        --num-gpus 1 [other args]

Resume:
    python train.py --config-file configs/mammo_dinov2_dino.py \\
        --resume [other args]
"""

import argparse
import logging
import os
import sys

# Ensure detrex projects are importable (set DETREX_ROOT env var or pass --detrex-root)
_default_detrex = os.environ.get("DETREX_ROOT", "/content/detrex")
sys.path.insert(0, _default_detrex)
sys.path.insert(0, os.path.dirname(__file__))

from detectron2.config import LazyConfig, instantiate
from detectron2.engine import (
    default_argument_parser,
    default_setup,
    default_writers,
    hooks,
    launch,
)
from detectron2.engine.defaults import create_ddp_model
from detectron2.evaluation import inference_on_dataset, print_csv_format
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.utils import comm

from rfdetr.data.registration import register_mammo_dataset

logger = logging.getLogger("mammo_train")


# ─── Detrex Trainer (copied from detrex/tools/train_net.py) ──────────────────
# We use it verbatim — this is detrex code, not a reimplementation.

import time
import torch
from torch.nn.parallel import DataParallel, DistributedDataParallel
from detectron2.engine import SimpleTrainer


class Trainer(SimpleTrainer):
    """Combines SimpleTrainer + AMP, adapted from detrex/tools/train_net.py."""

    def __init__(self, model, dataloader, optimizer, amp=False,
                 clip_grad_params=None, grad_scaler=None):
        super().__init__(model=model, data_loader=dataloader, optimizer=optimizer)
        unsupported = "AMPTrainer does not support single-process multi-device training!"
        if isinstance(model, DistributedDataParallel):
            assert not (model.device_ids and len(model.device_ids) > 1), unsupported
        assert not isinstance(model, DataParallel), unsupported
        if amp:
            if grad_scaler is None:
                from torch.cuda.amp import GradScaler
                grad_scaler = GradScaler()
        self.grad_scaler = grad_scaler
        self.amp = amp
        self.clip_grad_params = clip_grad_params

    def run_step(self):
        assert self.model.training
        start = time.perf_counter()
        data = next(self._data_loader_iter)
        data_time = time.perf_counter() - start

        with torch.cuda.amp.autocast(enabled=self.amp):
            loss_dict = self.model(data)
            losses = sum(loss_dict.values())

        self.optimizer.zero_grad()
        if self.amp:
            self.grad_scaler.scale(losses).backward()
            if self.clip_grad_params:
                self.grad_scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), **self.clip_grad_params
                )
            self.grad_scaler.step(self.optimizer)
            self.grad_scaler.update()
        else:
            losses.backward()
            if self.clip_grad_params:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), **self.clip_grad_params
                )
            self.optimizer.step()

        self._write_metrics(loss_dict, data_time)


# ─── Training Logic ────────────────────────────────────────────────────────────

def do_train(args, cfg):
    model = instantiate(cfg.model)
    model.to(cfg.train.device)
    model = create_ddp_model(model)

    optimizer = instantiate(cfg.optimizer, params=model.parameters())
    train_loader = instantiate(cfg.dataloader.train)

    trainer = Trainer(
        model=model,
        dataloader=train_loader,
        optimizer=optimizer,
        amp=cfg.train.amp.enabled,
        clip_grad_params=cfg.train.get("clip_grad", {}).get("params", None),
    )

    checkpointer = DetectionCheckpointer(
        model,
        cfg.train.output_dir,
        trainer=trainer,
        optimizer=optimizer,
    )

    trainer.register_hooks([
        hooks.IterationTimer(),
        hooks.LRScheduler(scheduler=instantiate(cfg.lr_multiplier)),
        hooks.PeriodicCheckpointer(checkpointer, cfg.train.checkpointer.period),
        hooks.EvalHook(
            cfg.train.eval_period,
            lambda: do_eval(cfg, model),
        ),
        hooks.PeriodicWriter(
            default_writers(cfg.train.output_dir, cfg.train.max_iter),
            period=cfg.train.log_period,
        ),
    ])

    start_iter = (
        checkpointer.resume_or_load(cfg.train.init_checkpoint, resume=args.resume).get(
            "iteration", -1
        ) + 1
        if args.resume or cfg.train.get("init_checkpoint")
        else 0
    )

    trainer.train(start_iter, cfg.train.max_iter)


def do_eval(cfg, model):
    from detectron2.evaluation import COCOEvaluator, inference_on_dataset, print_csv_format
    from detectron2.data import build_detection_test_loader
    from rfdetr.data.mapper import Mammo16BitMapper
    import detectron2.data.transforms as T

    test_loader = build_detection_test_loader(
        dataset=cfg.dataloader.test.dataset,
        mapper=Mammo16BitMapper(
            augmentation=[
                T.ResizeShortestEdge(
                    short_edge_length=(cfg.dataloader.get("test_size", 812),),
                    max_size=cfg.dataloader.get("max_size", 1624),
                    sample_style="choice",
                )
            ],
            augmentation_with_crop=None,
            is_train=False,
        ),
        num_workers=cfg.dataloader.test.num_workers,
    )
    evaluator = COCOEvaluator(
        cfg.dataloader.test.dataset,
        output_dir=os.path.join(cfg.train.output_dir, "eval"),
    )
    results = inference_on_dataset(model, test_loader, evaluator)
    print_csv_format(results)
    return results


# ─── CLI ──────────────────────────────────────────────────────────────────────

def build_arg_parser():
    parser = default_argument_parser()
    parser.add_argument("--train-json",      required=True, help="Path to COCO train JSON")
    parser.add_argument("--val-json",        required=True, help="Path to COCO val JSON")
    parser.add_argument("--images-dir",      required=True, help="Root directory for images")
    parser.add_argument("--dinov2-weights",  default=None,  help="Path to DINOv2 checkpoint .pth")
    parser.add_argument("--detrex-root",     default=_default_detrex,
                        help=f"Path to detrex clone (default: {_default_detrex})")
    return parser


def main(args):
    # Insert detrex into path (may differ from default)
    if args.detrex_root not in sys.path:
        sys.path.insert(0, args.detrex_root)

    # Register mammography datasets and auto-detect classes from JSON
    thing_classes = register_mammo_dataset(
        train_json=args.train_json,
        val_json=args.val_json,
        images_dir=args.images_dir,
    )
    num_classes = len(thing_classes)
    logger.info(f"Auto-detected {num_classes} class(es): {thing_classes}")

    cfg = LazyConfig.load(args.config_file)
    cfg = LazyConfig.apply_overrides(cfg, args.opts)

    # ── Auto-patch num_classes everywhere in the config ──────────────────────
    # This avoids any hardcoded num_classes=1 in the config file.
    cfg.model.num_classes = num_classes
    if hasattr(cfg.model, "criterion") and hasattr(cfg.model.criterion, "num_classes"):
        cfg.model.criterion.num_classes = num_classes
    logger.info(f"Config patched: model.num_classes = {num_classes}")

    # Inject DINOv2 weights path into config
    if args.dinov2_weights:
        cfg.model.backbone.backbone.checkpoint_path = args.dinov2_weights

    # Inject output dir
    if not cfg.train.get("output_dir"):
        cfg.train.output_dir = "./output"
    os.makedirs(cfg.train.output_dir, exist_ok=True)

    default_setup(cfg, args)

    if args.eval_only:
        model = instantiate(cfg.model)
        model.to(cfg.train.device)
        DetectionCheckpointer(model).load(cfg.train.init_checkpoint)
        results = do_eval(cfg, model)
        print_csv_format(results)
        return results

    do_train(args, cfg)


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
