"""rfdetr.data.mapper

Dataset mapper and Detectron2-compatible augmentations for 16-bit mammography images.

Features:
  - normalize_with_percentiles: Foreground-aware percentile intensity windowing.
  - LesionAwareCrop: Detectron2 Augmentation that guarantees 100% containment of target lesions.
  - Mammo16BitMapper: Custom detectron2 dataset mapper reading uint16 DICOM/PNG images.
"""

from __future__ import annotations

import copy
import logging
import os
import random
from typing import Any, Dict, List, Optional, Tuple, Union

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


# ─── Fallback Transform Classes (Used when Detectron2 is not installed) ──────

if T is not None and hasattr(T, "Augmentation"):
    _BaseAugmentation = T.Augmentation
    _CropTransform = T.CropTransform
    _NoOpTransform = T.NoOpTransform
    _TransformList = T.TransformList
    _AugInput = getattr(T, "AugInput", None)
else:
    class _Transform:
        def apply_image(self, img: np.ndarray) -> np.ndarray:
            return img

        def apply_coords(self, coords: np.ndarray) -> np.ndarray:
            return coords

        def apply_box(self, box: np.ndarray) -> np.ndarray:
            return box

    class _NoOpTransform(_Transform):
        pass

    class _CropTransform(_Transform):
        def __init__(self, x0: int, y0: int, w: int, h: int) -> None:
            self.x0 = int(x0)
            self.y0 = int(y0)
            self.w = int(w)
            self.h = int(h)

        def apply_image(self, img: np.ndarray) -> np.ndarray:
            return img[self.y0 : self.y0 + self.h, self.x0 : self.x0 + self.w]

        def apply_coords(self, coords: np.ndarray) -> np.ndarray:
            coords = np.asarray(coords).copy()
            coords[:, 0] -= self.x0
            coords[:, 1] -= self.y0
            return coords

        def apply_box(self, box: np.ndarray) -> np.ndarray:
            box = np.asarray(box).copy()
            box[:, [0, 2]] -= self.x0
            box[:, [1, 3]] -= self.y0
            box[:, [0, 2]] = np.clip(box[:, [0, 2]], 0, self.w)
            box[:, [1, 3]] = np.clip(box[:, [1, 3]], 0, self.h)
            return box

    class _TransformList(_Transform):
        def __init__(self, transforms: List[Any]) -> None:
            self.transforms = list(transforms)

        def apply_image(self, img: np.ndarray) -> np.ndarray:
            for t in self.transforms:
                img = t.apply_image(img)
            return img

        def apply_coords(self, coords: np.ndarray) -> np.ndarray:
            for t in self.transforms:
                coords = t.apply_coords(coords)
            return coords

        def apply_box(self, box: np.ndarray) -> np.ndarray:
            for t in self.transforms:
                box = t.apply_box(box)
            return box

    class _BaseAugmentation:
        def get_transform(self, *args: Any, **kwargs: Any) -> Any:
            raise NotImplementedError

    class _AugInput:
        def __init__(
            self,
            image: np.ndarray,
            boxes: Optional[np.ndarray] = None,
            sem_seg: Optional[np.ndarray] = None,
        ) -> None:
            self.image = image
            self.boxes = boxes
            self.sem_seg = sem_seg

        def transform(self, tf: Any) -> None:
            if self.image is not None:
                self.image = tf.apply_image(self.image)
            if self.boxes is not None:
                self.boxes = tf.apply_box(self.boxes)

CropTransform = _CropTransform
NoOpTransform = _NoOpTransform
TransformList = _TransformList
AugInput = _AugInput if _AugInput is not None else getattr(T, "AugInput", None)


# ─── LesionAwareCrop (Detectron2 Augmentation) ────────────────────────────────

class LesionAwareCrop(_BaseAugmentation):
    """Detectron2-idiomatic Augmentation that crops an image while guaranteeing

    that at least one lesion (bounding box) is 100% contained within the crop window.

    Detectron2 Augmentation Architecture:
        In Detectron2, `T.Augmentation` defines a stochastic policy via `get_transform`.
        When invoked via `aug(input)` on a `T.AugInput(image, boxes=boxes)`, Detectron2
        inspects `get_transform(self, image, boxes=None)` by parameter name and passes
        both the image and bounding boxes dynamically.

    Mathematical Guarantee:
        Given crop size (cw, ch) and a chosen lesion [bx, by, bw, bh]:
        - If bw <= cw: x_min = max(0, bx + bw - cw), x_max = min(W - cw, bx)
          Any x0 in [x_min, x_max] satisfies: x0 <= bx and bx + bw <= x0 + cw.
        - If bh <= ch: y_min = max(0, by + bh - ch), y_max = min(H - ch, by)
          Any y0 in [y_min, y_max] satisfies: y0 <= by and by + bh <= y0 + ch.
        - If lesion is larger than crop size, centers the crop on the lesion center.
        - If no lesions exist, samples a uniformly random valid crop box.

    Args:
        crop_size: Tuple (height, width) of the crop window (default: (518, 518)).
        prob: Probability of applying the crop (default: 0.0). When not applied,
            returns NoOpTransform().
    """

    def __init__(
        self,
        crop_size: Tuple[int, int] = (518, 518),
        prob: float = 0.0,
    ) -> None:
        if hasattr(super(), "__init__"):
            try:
                super().__init__()
            except TypeError:
                pass
        self.crop_size = (int(crop_size[0]), int(crop_size[1]))
        self.prob = float(prob)

    def get_crop_box(
        self,
        image_shape: Tuple[int, int],
        boxes: Optional[Union[np.ndarray, List[Any]]] = None,
    ) -> Tuple[int, int, int, int]:
        """Compute (x0, y0, cw, ch) guaranteed to contain a lesion if boxes exist."""
        H, W = image_shape[:2]
        ch = min(H, self.crop_size[0])
        cw = min(W, self.crop_size[1])

        target = None
        if boxes is not None:
            if isinstance(boxes, np.ndarray) and boxes.size > 0:
                valid_boxes = []
                for b in boxes:
                    x1, y1, x2, y2 = b[:4]
                    if x2 > x1 and y2 > y1:
                        valid_boxes.append((float(x1), float(y1), float(x2 - x1), float(y2 - y1)))
                if valid_boxes:
                    target = random.choice(valid_boxes)
            elif isinstance(boxes, (list, tuple)) and len(boxes) > 0:
                valid_boxes = []
                for item in boxes:
                    if isinstance(item, dict):
                        if item.get("iscrowd", 0) == 0 and "bbox" in item:
                            bb = item["bbox"]
                            if len(bb) >= 4 and bb[2] > 0 and bb[3] > 0:
                                valid_boxes.append((float(bb[0]), float(bb[1]), float(bb[2]), float(bb[3])))
                    elif isinstance(item, (list, tuple, np.ndarray)) and len(item) >= 4:
                        x1, y1, x2, y2 = item[:4]
                        if x2 > x1 and y2 > y1:
                            valid_boxes.append((float(x1), float(y1), float(x2 - x1), float(y2 - y1)))
                if valid_boxes:
                    target = random.choice(valid_boxes)

        if target is not None:
            bx, by, bw, bh = target

            if bw <= cw:
                x_min = max(0, int(np.floor(bx + bw - cw)))
                x_max = min(int(np.floor(W - cw)), int(np.floor(bx)))
                x0 = random.randint(x_min, max(x_min, x_max))
            else:
                cx = int(bx + bw / 2.0)
                x0 = int(np.clip(cx - cw // 2, 0, max(0, W - cw)))

            if bh <= ch:
                y_min = max(0, int(np.floor(by + bh - ch)))
                y_max = min(int(np.floor(H - ch)), int(np.floor(by)))
                y0 = random.randint(y_min, max(y_min, y_max))
            else:
                cy = int(by + bh / 2.0)
                y0 = int(np.clip(cy - ch // 2, 0, max(0, H - ch)))
        else:
            x0 = random.randint(0, max(0, W - cw)) if W > cw else 0
            y0 = random.randint(0, max(0, H - ch)) if H > ch else 0

        return int(x0), int(y0), int(cw), int(ch)

    def get_transform(
        self,
        image: np.ndarray,
        boxes: Optional[Union[np.ndarray, List[Any]]] = None,
    ) -> Any:
        """Detectron2 get_transform implementation.

        Returns a CropTransform guaranteed to contain at least one lesion,
        or NoOpTransform if probability check fails or prob == 0.0.
        """
        if self.prob <= 0.0 or random.random() >= self.prob:
            return _NoOpTransform()
        x0, y0, cw, ch = self.get_crop_box(image.shape[:2], boxes)
        return _CropTransform(x0, y0, cw, ch)

    def __call__(self, aug_input: Any) -> Any:
        """Execute augmentation on an AugInput instance (fallback when not subclassing D2 Augmentation)."""
        if hasattr(super(), "__call__"):
            return super().__call__(aug_input)
        image = getattr(aug_input, "image", None)
        boxes = getattr(aug_input, "boxes", None)
        if image is None and isinstance(aug_input, np.ndarray):
            image = aug_input
        tf = self.get_transform(image, boxes=boxes)
        if hasattr(aug_input, "transform"):
            aug_input.transform(tf)
        return tf


# ─── Mammo16BitMapper ────────────────────────────────────────────────────────

class Mammo16BitMapper:
    """Dataset mapper for 16-bit mammography images (DICOM-exported PNG/TIF).

    Follows the same interface as DetrDatasetMapper:
      - Takes a detectron2 dataset dict
      - Returns a dict with 'image' (C, H, W) float32 tensor in [0.0, 1.0] and 'instances'

    Differences from standard mappers:
      - Reads uint16 images via OpenCV (IMREAD_UNCHANGED)
      - Applies foreground percentile intensity windowing before augmentation
      - Replicates grayscale to 3 channels for DINOv2 backbone
      - Outputs float32 [0.0, 1.0] preserving full 16-bit dynamic range
      - Integrates LesionAwareCrop ensuring small lesions are never cropped away

    Args:
        augmentation: List of detectron2 augmentation transforms.
        augmentation_with_crop: Optional legacy list with crop.
        is_train: Training mode — loads annotations when True.
        mask_on: Whether to keep segmentation masks (default: False).
        low_pct: Lower percentile for intensity windowing (default: 1.0).
        high_pct: Upper percentile for intensity windowing (default: 99.0).
        images_fallback_dir: Optional directory to search if file_name not found.
        use_percentile_norm: Whether to use foreground percentile normalization (default: True).
        crop_prob: Probability of applying lesion-aware crop (default: 0.0).
        crop_size: (H, W) crop window size (default: (518, 518)).
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
        self.crop_aug = LesionAwareCrop(crop_size=self.crop_size, prob=self.crop_prob)
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
        """Compute (x0, y0, cw, ch) guaranteed to contain a lesion if annotations exist.

        Delegates to LesionAwareCrop.get_crop_box for unified logic.
        """
        return self.crop_aug.get_crop_box(image_shape, annotations)

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

        # 1. Lesion-aware crop (if enabled with crop_prob > 0)
        crop_tf = None
        if self.crop_prob > 0.0:
            crop_tf = self.crop_aug.get_transform(
                image,
                boxes=dataset_dict.get("annotations", []),
            )
            if not isinstance(crop_tf, (_NoOpTransform, getattr(T, "NoOpTransform", _NoOpTransform))):
                image = crop_tf.apply_image(image)

        # 2. Sequential augmentations (ResizeShortestEdge, RandomFlip, etc.)
        image, other_transforms = T.apply_transform_gens(self.augmentation, image)

        # 3. Combine transforms so box coordinates are updated accordingly
        if crop_tf is not None and not isinstance(crop_tf, (_NoOpTransform, getattr(T, "NoOpTransform", _NoOpTransform))):
            sub_tfs = other_transforms.transforms if hasattr(other_transforms, "transforms") else [other_transforms]
            transforms = T.TransformList([crop_tf] + list(sub_tfs))
        else:
            transforms = other_transforms

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
