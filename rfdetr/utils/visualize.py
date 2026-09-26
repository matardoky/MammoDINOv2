"""rfdetr.utils.visualize

Dataset and prediction visualization utilities for 16-bit mammography images.

Uses detectron2's Visualizer for annotation rendering, but replaces the
standard cv2.imread with our percentile-normalized 16-bit reader so the
images display correctly (otherwise uint16 would appear all black in matplotlib).
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

try:
    from detectron2.data import DatasetCatalog, MetadataCatalog
    from detectron2.utils.visualizer import Visualizer
except ImportError:
    DatasetCatalog = None
    MetadataCatalog = None
    Visualizer = None


def read_mammo_uint8(
    file_name: str,
    low_pct: float = 1.0,
    high_pct: float = 99.0,
    images_fallback_dir: Optional[str] = None,
) -> np.ndarray:
    """Read a 16-bit mammography image and return uint8 RGB (H, W, 3).

    Applies percentile windowing so the image is visible in matplotlib.
    Falls back to standard read for 8-bit images.
    """
    import os

    path = file_name
    if not os.path.isfile(path) and images_fallback_dir:
        candidate = os.path.join(images_fallback_dir, os.path.basename(file_name))
        if os.path.isfile(candidate):
            path = candidate

    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise IOError(f"Could not read: {path}")

    if img.ndim == 3:
        # Already multi-channel BGR 8-bit
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Grayscale — percentile normalization
    lo = float(np.percentile(img, low_pct))
    hi = float(np.percentile(img, high_pct))
    if hi <= lo:
        hi = lo + 1.0
    img_norm = np.clip((img.astype(np.float32) - lo) / (hi - lo), 0.0, 1.0)
    img_u8 = (img_norm * 255).astype(np.uint8)
    return np.stack([img_u8, img_u8, img_u8], axis=-1)  # (H, W, 3) RGB


def visualize_dataset(
    dataset_name: str,
    num_images: int = 3,
    low_pct: float = 1.0,
    high_pct: float = 99.0,
    scale: float = 1.0,
    save_dir: Optional[str] = None,
    seed: Optional[int] = None,
    images_fallback_dir: Optional[str] = None,
) -> None:
    """Visualize random samples from a registered detectron2 dataset.

    Shows original image (percentile-normalized) alongside the image
    with ground-truth annotations drawn by detectron2's Visualizer.

    Args:
        dataset_name: Name of a registered DatasetCatalog dataset.
        num_images: Number of random samples to display (default: 3).
        low_pct: Lower percentile for 16-bit windowing (default: 1.0).
        high_pct: Upper percentile for 16-bit windowing (default: 99.0).
        scale: Visualizer drawing scale (default: 1.0).
        save_dir: If provided, save figures to this directory instead of showing.
        seed: Optional random seed for reproducibility.
        images_fallback_dir: Optional fallback directory for locating images.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        raise ImportError("matplotlib is required for visualization: pip install matplotlib")

    if seed is not None:
        random.seed(seed)

    dataset_dicts = DatasetCatalog.get(dataset_name)
    metadata = MetadataCatalog.get(dataset_name)

    samples = random.sample(dataset_dicts, min(num_images, len(dataset_dicts)))

    fig, axs = plt.subplots(
        nrows=len(samples),
        ncols=2,
        figsize=(15, 7 * len(samples)),
        squeeze=False,
    )

    for i, d in enumerate(samples):
        # Read with our 16-bit normalizer instead of raw cv2.imread
        img_rgb = read_mammo_uint8(
            d["file_name"],
            low_pct=low_pct,
            high_pct=high_pct,
            images_fallback_dir=images_fallback_dir,
        )

        # detectron2 Visualizer expects RGB
        visualizer = Visualizer(img_rgb, metadata=metadata, scale=scale)
        out = visualizer.draw_dataset_dict(d)

        # Left: original (normalized)
        axs[i, 0].imshow(img_rgb)
        axs[i, 0].axis("off")
        axs[i, 0].set_title(
            f"Original — {Path(d['file_name']).name}\n"
            f"({img_rgb.shape[1]}×{img_rgb.shape[0]} px)"
        )

        # Right: with annotations
        axs[i, 1].imshow(out.get_image())
        axs[i, 1].axis("off")
        n_ann = len(d.get("annotations", []))
        axs[i, 1].set_title(f"Annotations ({n_ann} instance{'s' if n_ann != 1 else ''})")

    plt.suptitle(f"Dataset: {dataset_name}", fontsize=14, fontweight="bold", y=1.01)
    plt.tight_layout()

    if save_dir:
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        out_path = Path(save_dir) / f"viz_{dataset_name}.png"
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {out_path}")
    else:
        plt.show()

    plt.close(fig)
