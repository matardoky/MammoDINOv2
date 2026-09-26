"""rfdetr.data

Custom data components for 16-bit mammography images.

What we implement:
  - Mammo16BitMapper: reads uint16 DICOM/PNG, normalizes with percentile windowing,
    converts to 3-channel float32 for detectron2/detrex.

What comes from detectron2/detrex (used directly, not reimplemented):
  - DatasetCatalog / MetadataCatalog registration
  - build_detection_train_loader / build_detection_test_loader
  - detectron2.data.transforms (augmentations)
  - COCOEvaluator (via detectron2.evaluation)
"""

from rfdetr.data.mapper import Mammo16BitMapper
from rfdetr.data.registration import register_mammo_dataset

__all__ = ["Mammo16BitMapper", "register_mammo_dataset"]
