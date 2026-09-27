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
from typing import Any, Dict, List, Optional


# Ensure detrex projects are importable (set DETREX_ROOT env var or pass --detrex-root)
_local_detrex = os.path.join(os.path.dirname(os.path.abspath(__file__)), "detrex")
_default_detrex = os.environ.get(
    "DETREX_ROOT",
    _local_detrex if os.path.isdir(_local_detrex) else "/content/detrex"
)
sys.path.insert(0, _default_detrex)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

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
from rfdetr.utils.amp_patch import patch_detrex_ms_deform_attn, patch_detrex_source_file

logger = logging.getLogger("mammo_train")



# ─── Detrex Trainer with Gradient Accumulation & AMP bf16/fp16 ───────────────

from contextlib import nullcontext
import time
import torch
from torch.nn.parallel import DataParallel, DistributedDataParallel
from detectron2.engine import SimpleTrainer


class Trainer(SimpleTrainer):
    """Combines SimpleTrainer + AMP (bf16/fp16) + Gradient Accumulation + Grad Norm Logging.

    Adapted from detrex/tools/train_net.py with single-GPU and Colab optimizations:
      - Gradient accumulation over N batches (effective batch size = physical batch * N).
      - bfloat16 native execution without unnecessary GradScaler.
      - float16 execution with GradScaler when bfloat16 is unsupported.
      - Logging of the true L2 gradient norm to EventStorage/TensorBoard.
    """

    def __init__(
        self,
        model,
        dataloader,
        optimizer,
        amp: bool = False,
        amp_dtype: torch.dtype = torch.float16,
        clip_grad_params: Optional[dict] = None,
        grad_scaler=None,
        grad_accum_steps: int = 1,
    ):
        super().__init__(model=model, data_loader=dataloader, optimizer=optimizer)
        unsupported = "AMPTrainer does not support single-process multi-device training!"
        if isinstance(model, DistributedDataParallel):
            assert not (model.device_ids and len(model.device_ids) > 1), unsupported
        assert not isinstance(model, DataParallel), unsupported

        self.amp = amp
        self.amp_dtype = amp_dtype
        self.clip_grad_params = clip_grad_params
        self.grad_accum_steps = max(1, int(grad_accum_steps))

        # GradScaler is ONLY required for float16 to prevent underflow.
        # bfloat16 shares the 8-bit exponent of float32 and does not need scaling.
        self.grad_scaler = None
        if amp and (self.amp_dtype == torch.float16):
            if grad_scaler is None:
                if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
                    grad_scaler = torch.amp.GradScaler("cuda")
                else:
                    from torch.cuda.amp import GradScaler
                    grad_scaler = GradScaler()
            self.grad_scaler = grad_scaler

    def _should_step(self) -> bool:
        """Step optimizer every grad_accum_steps or at the final iteration."""
        it = self.iter + 1
        max_iter = getattr(self, "max_iter", None)
        if max_iter is not None and it >= max_iter:
            return True
        return it % self.grad_accum_steps == 0

    def clip_grads(self, params) -> Optional[torch.Tensor]:
        params = [p for p in params if p.requires_grad and p.grad is not None]
        if params and self.clip_grad_params is not None:
            norm = torch.nn.utils.clip_grad_norm_(params, **self.clip_grad_params)
            if hasattr(self, "storage") and self.storage is not None:
                self.storage.put_scalar("grad_norm", norm.item())
            return norm
        return None

    def run_step(self):
        assert self.model.training
        start = time.perf_counter()
        data = next(self._data_loader_iter)
        data_time = time.perf_counter() - start

        accum = self.grad_accum_steps
        do_step = self._should_step()

        # Context manager for autocast
        autocast_ctx = (
            torch.amp.autocast("cuda", dtype=self.amp_dtype)
            if (self.amp and torch.cuda.is_available())
            else nullcontext()
        )

        with autocast_ctx:
            loss_dict = self.model(data)
            if isinstance(loss_dict, torch.Tensor):
                loss_dict = {"total_loss": loss_dict}
            losses = sum(loss_dict.values()) / accum

        if self.grad_scaler is not None:
            # float16 with loss scaler
            self.grad_scaler.scale(losses).backward()
            if do_step:
                if self.clip_grad_params:
                    self.grad_scaler.unscale_(self.optimizer)
                    self.clip_grads(self.model.parameters())
                self.grad_scaler.step(self.optimizer)
                self.grad_scaler.update()
                self.optimizer.zero_grad()
        else:
            # bfloat16 or float32 (no scaler needed)
            losses.backward()
            if do_step:
                if self.clip_grad_params:
                    self.clip_grads(self.model.parameters())
                self.optimizer.step()
                self.optimizer.zero_grad()

        # Write true unscaled micro-batch losses for dashboard logging
        loss_log = {
            k: (v.detach() * accum if isinstance(v, torch.Tensor) else v)
            for k, v in loss_dict.items()
        }
        self._write_metrics(loss_log, data_time)

    def state_dict(self):
        ret = super().state_dict()
        if self.grad_scaler is not None:
            ret["grad_scaler"] = self.grad_scaler.state_dict()
        return ret

    def load_state_dict(self, sd):
        super().load_state_dict(sd)
        if self.grad_scaler is not None and "grad_scaler" in sd:
            self.grad_scaler.load_state_dict(sd["grad_scaler"])


# ─── Training Logic ────────────────────────────────────────────────────────────

def do_train(args, cfg):
    device = cfg.train.device if (torch.cuda.is_available() and cfg.train.device == "cuda") else "cpu"
    if device != cfg.train.device:
        logger.warning(f"CUDA not available — falling back to device='{device}'")
    cfg.train.device = device
    cfg.model.device = device

    model = instantiate(cfg.model)
    model.to(cfg.train.device)
    model = create_ddp_model(model)

    if hasattr(cfg.optimizer, "params"):
        cfg.optimizer.params.model = model
    else:
        cfg.optimizer.params = model.parameters()
    optimizer = instantiate(cfg.optimizer)

    train_loader = instantiate(cfg.dataloader.train)

    # Determine mixed precision dtype (Detrex DINO canonical is float16 with GradScaler)
    amp_enabled = cfg.train.amp.enabled and (device == "cuda")
    amp_dtype_str = getattr(args, "amp_dtype", None) or cfg.train.amp.get("dtype", "float16")
    if amp_dtype_str in ("auto", "float16"):
        amp_dtype = torch.float16
    elif amp_dtype_str == "bfloat16":
        amp_dtype = torch.bfloat16
    else:
        amp_dtype = torch.float32

    # Determine gradient accumulation steps (CLI flag overrides config)
    grad_accum_steps = (
        getattr(args, "accum_steps", None)
        if getattr(args, "accum_steps", None) is not None
        else cfg.train.get("grad_accum_steps", 1)
    )

    trainer = Trainer(
        model=model,
        dataloader=train_loader,
        optimizer=optimizer,
        amp=amp_enabled,
        amp_dtype=amp_dtype,
        clip_grad_params=cfg.train.get("clip_grad", {}).get("params", None),
        grad_accum_steps=grad_accum_steps,
    )

    lr_scheduler = instantiate(cfg.lr_multiplier)

    extra_checkpointables = {"trainer": trainer, "optimizer": optimizer}
    if hasattr(lr_scheduler, "state_dict"):
        extra_checkpointables["scheduler"] = lr_scheduler

    checkpointer = DetectionCheckpointer(
        model,
        cfg.train.output_dir,
        **extra_checkpointables,
    )


    eval_hook = hooks.EvalHook(
        cfg.train.eval_period,
        lambda: do_eval(cfg, model),
        eval_after_train=True,  # always evaluate at end of training
    )

    all_hooks = [
        hooks.IterationTimer(),
        hooks.LRScheduler(scheduler=lr_scheduler),
        # ── Periodic checkpoint (only main process in multi-GPU) ───────────────
        hooks.PeriodicCheckpointer(checkpointer, cfg.train.checkpointer.period)
        if comm.is_main_process()
        else None,
        eval_hook,
        # ── Best model checkpoint (only main process in multi-GPU) ────────────
        hooks.BestCheckpointer(
            cfg.train.eval_period,
            checkpointer,
            val_metric="bbox/AP50",
            mode="max",
            file_prefix="model_best",
        )
        if comm.is_main_process()
        else None,
        # ── Periodic log / tensorboard writer ─────────────────────────────────
        hooks.PeriodicWriter(
            default_writers(cfg.train.output_dir, cfg.train.max_iter),
            period=cfg.train.log_period,
        )
        if comm.is_main_process()
        else None,
    ]
    trainer.register_hooks([h for h in all_hooks if h is not None])

    start_iter = (
        checkpointer.resume_or_load(cfg.train.init_checkpoint, resume=args.resume).get(
            "iteration", -1
        ) + 1
        if args.resume or cfg.train.get("init_checkpoint")
        else 0
    )

    trainer.train(start_iter, cfg.train.max_iter)


def do_eval(cfg, model):
    if "evaluator" in cfg.dataloader and "test" in cfg.dataloader:
        test_loader = instantiate(cfg.dataloader.test)
        evaluator = instantiate(cfg.dataloader.evaluator)
    else:
        from detectron2.data import DatasetCatalog, build_detection_test_loader
        from detectron2.evaluation import COCOEvaluator
        from rfdetr.data.mapper import Mammo16BitMapper
        import detectron2.data.transforms as T

        test_loader = build_detection_test_loader(
            dataset=DatasetCatalog.get("mammo_val"),
            mapper=Mammo16BitMapper(
                augmentation=[
                    T.ResizeShortestEdge(
                        short_edge_length=(812,),
                        max_size=1624,
                        sample_style="choice",
                    )
                ],
                augmentation_with_crop=None,
                is_train=False,
            ),
            num_workers=2,
        )
        evaluator = COCOEvaluator(
            dataset_name="mammo_val",
            output_dir=os.path.join(cfg.train.output_dir, "eval"),
        )

    results = inference_on_dataset(model, test_loader, evaluator)
    print_csv_format(results)
    return results


# ─── CLI ──────────────────────────────────────────────────────────────────────

def build_arg_parser():
    parser = default_argument_parser()
    parser.add_argument("--output-dir",      default=None,  help="Path to output directory for checkpoints and logs")
    parser.add_argument("--train-json",      required=True, help="Path to COCO train JSON")
    parser.add_argument("--val-json",        required=True, help="Path to COCO val JSON")
    parser.add_argument("--images-dir",      required=True, help="Root directory for images")
    parser.add_argument("--dinov2-weights",  default=None,  help="Path to DINOv2 checkpoint .pth")
    parser.add_argument("--accum-steps",     type=int, default=None,
                        help="Gradient accumulation steps (default: from config or 1)")
    parser.add_argument("--amp-dtype",       choices=["auto", "bfloat16", "float16", "float32"], default="float16",
                        help="Mixed precision dtype: float16 (default, Detrex standard), bfloat16, float32, or auto")
    parser.add_argument("--detrex-root",     default=_default_detrex,
                        help=f"Path to detrex clone (default: {_default_detrex})")
    return parser


def main(args):
    # Insert detrex into path (may differ from default)
    if args.detrex_root not in sys.path:
        sys.path.insert(0, args.detrex_root)

    # Patch Detrex MultiScaleDeformableAttn to ensure seamless fp16/bf16 CUDA execution
    patch_detrex_source_file(args.detrex_root)
    patch_detrex_ms_deform_attn()

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
    if getattr(args, "output_dir", None):
        cfg.train.output_dir = args.output_dir
    elif not cfg.train.get("output_dir"):
        cfg.train.output_dir = "./output"
    os.makedirs(cfg.train.output_dir, exist_ok=True)

    # Inject images fallback dir into mappers
    if hasattr(cfg, "dataloader"):
        if hasattr(cfg.dataloader, "train") and hasattr(cfg.dataloader.train, "mapper"):
            cfg.dataloader.train.mapper.images_fallback_dir = args.images_dir
        if hasattr(cfg.dataloader, "test") and hasattr(cfg.dataloader.test, "mapper"):
            cfg.dataloader.test.mapper.images_fallback_dir = args.images_dir

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
