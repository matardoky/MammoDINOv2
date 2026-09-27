"""rfdetr.data.mapper

Dataset mapper for 16-bit mammography images.

Implements the same interface as DetrDatasetMapper but handles uint16
DICOM-exported images that standard detectron2/detrex mappers cannot read.

Why not subclass DetrDatasetMapper: it has no _load_image hook — image
loading is inlined in __call__, so subclassing would require duplicating
the entire method. A standalone mapper is simpler and more robust.
"""

from __future__ import annotations

import copy
import logging
import os
import random
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch

try:
    import detectron2.data.transforms as T
    from detectron2.data import detection_utils as utils
except ImportError:
    T = None
    utils = None

logger = logging.getLogger(__name__)


def normalize_with_percentiles(
    img_16bit: np.ndarray,
    low_pct: float = 1.0,
    high_pct: float = 99.0,
) -> np.ndarray:
    """Normalize 16-bit mammography image using foreground percentiles.

    Filters out black background air (pixels == 0) so percentiles reflect
    actual breast parenchyma. Returns float32 array in [0.0, 1.0].
    """
    if img_16bit is None or img_16bit.size == 0:
        return np.zeros_like(img_16bit, dtype=np.float32)
    h, w = img_16bit.shape[:2]
    step = max(1, min(h, w) // 300)
    sample = img_16bit[::step, ::step]
    fg = sample[sample > 0]
    target = fg if len(fg) >= 1000 else img_16bit[img_16bit > 0]
    if len(target) == 0:
        return np.zeros_like(img_16bit, dtype=np.float32)
    p_low, p_high = np.percentile(target, (low_pct, high_pct))
    if p_high <= p_low:
        return np.zeros_like(img_16bit, dtype=np.float32)
    return np.clip((img_16bit.astype(np.float32) - p_low) / (p_high - p_low), 0.0, 1.0).astype(np.float32)


class Mammo16BitMapper:
    """Dataset mapper for 16-bit mammography images (DICOM-exported PNG/TIF).

    Follows the same interface as DetrDatasetMapper:
      - Takes a detectron2 dataset dict
      - Returns a dict with 'image' (C, H, W) float32 tensor in [0.0, 1.0] and 'instances'

    Differences from the standard mapper:
      - Reads uint16 images via OpenCV (IMREAD_UNCHANGED)
      - Applies foreground percentile intensity windowing before augmentation
      - Replicates grayscale to 3 channels for DINOv2 backbone
      - Outputs float32 [0.0, 1.0] preserving full 16-bit dynamic range

    Args:
        augmentation: List of detectron2 augmentation transforms.
        augmentation_with_crop: Optional list with crop (randomly applied 50/50).
        is_train: Training mode — loads annotations when True.
        mask_on: Whether to keep segmentation masks (default: False).
        low_pct: Lower percentile for intensity windowing (default: 1.0).
        high_pct: Upper percentile for intensity windowing (default: 99.0).
        images_fallback_dir: Optional directory to search if file_name not found.
        use_percentile_norm: Whether to use foreground percentile normalization (default: True).
    """

    def __init__(
        self,
        augmentation: List[Any],
        augmentation_with_crop: Optional[List[Any]] = None,
        is_train: bool = True,
        mask_on: bool = False,
        low_pct: float = 1.0,
        high_pct: float = 99.0,
        images_fallback_dir: Optional[str] = None,
        use_percentile_norm: bool = True,
        crop_prob: float = 0.0,
        crop_size: Tuple[int, int] = (518, 518),
    ) -> None:
        self.augmentation = augmentation
        self.augmentation_with_crop = augmentation_with_crop
        self.crop_prob = float(crop_prob)
        self.crop_size = tuple(crop_size)
        self.is_train = is_train
        self.mask_on = mask_on
        self.low_pct = low_pct
        self.high_pct = high_pct
        self.images_fallback_dir = images_fallback_dir
        self.use_percentile_norm = use_percentile_norm

    def _resolve_path(self, file_name: str) -> str:
        if os.path.isfile(file_name):
            return file_name
        if self.images_fallback_dir:
            candidate = os.path.join(self.images_fallback_dir, os.path.basename(file_name))
            if os.path.isfile(candidate):
                return candidate
        raise FileNotFoundError(f"Image not found: {file_name}")

    def _compute_lesion_aware_crop_box(
        self,
        image_shape: Tuple[int, int],
        annotations: List[Dict[str, Any]],
    ) -> Tuple[int, int, int, int]:
        """Compute (x0, y0, cw, ch) guaranteed to contain a lesion if annotations exist."""
        H, W = image_shape[:2]
        cw = min(W, self.crop_size[1])
        ch = min(H, self.crop_size[0])

        valid_annos = [a for a in annotations if a.get("iscrowd", 0) == 0 and "bbox" in a]
        if valid_annos:
            target = random.choice(valid_annos)
            bx, by, bw, bh = target["bbox"]

            if bw <= cw:
                x_min = max(0, int(bx + bw - cw))
                x_max = min(int(W - cw), int(bx))
                x0 = random.randint(x_min, max(x_min, x_max))
            else:
                cx = int(bx + bw / 2)
                x0 = int(np.clip(cx - cw // 2, 0, max(0, W - cw)))

            if bh <= ch:
                y_min = max(0, int(by + bh - ch))
                y_max = min(int(H - ch), int(by))
                y0 = random.randint(y_min, max(y_min, y_max))
            else:
                cy = int(by + bh / 2)
                y0 = int(np.clip(cy - ch // 2, 0, max(0, H - ch)))
        else:
            x0 = random.randint(0, max(0, W - cw))
            y0 = random.randint(0, max(0, H - ch))

        return x0, y0, cw, ch

    def _read_image(self, file_name: str) -> np.ndarray:
        """Read 16-bit or 8-bit image -> float32 RGB (H, W, 3) in [0.0, 1.0]."""
        path = self._resolve_path(file_name)
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if img is None:
            raise IOError(f"OpenCV could not read: {path}")

        if img.dtype == np.uint16:
            if self.use_percentile_norm:
                img = normalize_with_percentiles(img, low_pct=self.low_pct, high_pct=self.high_pct)
            else:
                img = img.astype(np.float32) / 65535.0
        else:
            img = img.astype(np.float32) / 255.0

        if img.ndim == 2:
            img = np.stack([img] * 3, axis=-1)
        elif img.shape[2] == 1:
            img = np.repeat(img, 3, axis=2)
        elif img.shape[2] == 3 and img.dtype != np.float32:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0

        return img.astype(np.float32)

    def __call__(self, dataset_dict: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        if utils is None or T is None:
            raise ImportError("detectron2 is required to apply augmentations and instance transformations in __call__")
        dataset_dict = copy.deepcopy(dataset_dict)
        image = self._read_image(dataset_dict["file_name"])
        utils.check_image_size(dataset_dict, image)

        do_crop = (
            self.crop_prob > 0.0
            and float(torch.rand(1).item()) < self.crop_prob
        )
        if do_crop:
            x0, y0, cw, ch = self._compute_lesion_aware_crop_box(
                image.shape[:2],
                dataset_dict.get("annotations", []),
            )
            crop_tf = T.CropTransform(x0, y0, cw, ch)
            image = crop_tf.apply_image(image)
            image, other_transforms = T.apply_transform_gens(self.augmentation, image)
            sub_tfs = other_transforms.transforms if hasattr(other_transforms, "transforms") else [other_transforms]
            transforms = T.TransformList([crop_tf] + list(sub_tfs))
        else:
            image, transforms = T.apply_transform_gens(self.augmentation, image)

        image_shape = image.shape[:2]  # (H, W)
        dataset_dict["image"] = torch.as_tensor(np.ascontiguousarray(image.transpose(2, 0, 1)))

        if not self.is_train:
            dataset_dict.pop("annotations", None)
            return dataset_dict

        if "annotations" in dataset_dict:
            for anno in dataset_dict["annotations"]:
                if not self.mask_on:
                    anno.pop("segmentation", None)
                anno.pop("keypoints", None)

            annos = [
                utils.transform_instance_annotations(obj, transforms, image_shape)
                for obj in dataset_dict.pop("annotations")
                if obj.get("iscrowd", 0) == 0
            ]
            instances = utils.annotations_to_instances(annos, image_shape)
            dataset_dict["instances"] = utils.filter_empty_instances(instances)

        return dataset_dict
