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
from typing import Any, Dict, List, Optional

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


class Mammo16BitMapper:
    """Dataset mapper for 16-bit mammography images (DICOM-exported PNG/TIF).

    Follows the same interface as DetrDatasetMapper:
      - Takes a detectron2 dataset dict
      - Returns a dict with 'image' (C, H, W) tensor and 'instances'

    Differences from the standard mapper:
      - Reads uint16 images via OpenCV (IMREAD_UNCHANGED)
      - Applies percentile intensity windowing before augmentation
      - Replicates grayscale to 3 channels for DINOv2 backbone

    Args:
        augmentation: List of detectron2 augmentation transforms.
        augmentation_with_crop: Optional list with crop (randomly applied 50/50).
        is_train: Training mode — loads annotations when True.
        mask_on: Whether to keep segmentation masks (default: False).
        low_pct: Lower percentile for intensity windowing (default: 1.0).
        high_pct: Upper percentile for intensity windowing (default: 99.0).
        images_fallback_dir: Optional directory to search if file_name not found.
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
    ) -> None:
        self.augmentation = augmentation
        self.augmentation_with_crop = augmentation_with_crop
        self.is_train = is_train
        self.mask_on = mask_on
        self.low_pct = low_pct
        self.high_pct = high_pct
        self.images_fallback_dir = images_fallback_dir

    def _resolve_path(self, file_name: str) -> str:
        if os.path.isfile(file_name):
            return file_name
        if self.images_fallback_dir:
            candidate = os.path.join(self.images_fallback_dir, os.path.basename(file_name))
            if os.path.isfile(candidate):
                return candidate
        raise FileNotFoundError(f"Image not found: {file_name}")

    def _read_image(self, file_name: str) -> np.ndarray:
        """Read uint16 image → percentile normalization → uint8 RGB (H, W, 3)."""
        path = self._resolve_path(file_name)
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if img is None:
            raise IOError(f"OpenCV could not read: {path}")

        if img.ndim == 3:
            # Already multi-channel (8-bit) — standard BGR→RGB
            return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Grayscale uint8 or uint16 — percentile windowing
        lo = float(np.percentile(img, self.low_pct))
        hi = float(np.percentile(img, self.high_pct))
        if hi <= lo:
            hi = lo + 1.0
        img_norm = np.clip((img.astype(np.float32) - lo) / (hi - lo), 0.0, 1.0)
        img_u8 = (img_norm * 255).astype(np.uint8)
        return np.stack([img_u8, img_u8, img_u8], axis=-1)  # (H, W, 3) uint8

    def __call__(self, dataset_dict: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        if utils is None or T is None:
            raise ImportError("detectron2 is required to apply augmentations and instance transformations in __call__")
        dataset_dict = copy.deepcopy(dataset_dict)
        image = self._read_image(dataset_dict["file_name"])
        utils.check_image_size(dataset_dict, image)

        if self.augmentation_with_crop is None or np.random.rand() > 0.5:
            image, transforms = T.apply_transform_gens(self.augmentation, image)
        else:
            image, transforms = T.apply_transform_gens(self.augmentation_with_crop, image)

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
