#!/usr/bin/env python
"""visualize.py — Visualize mammography dataset annotations.

Usage (Colab / terminal):
    python visualize.py \\
        --train-json  /content/drive/MyDrive/EMBED_Dataset/curated/full_dataset/coco/mass_train.json \\
        --val-json    /content/drive/MyDrive/EMBED_Dataset/curated/full_dataset/coco/mass_val.json \\
        --images-dir  /content/mammo_data/images \\
        --split       train \\
        --num-images  4 \\
        --save-dir    ./viz_output

Or from a notebook:
    from rfdetr.data.registration import register_mammo_dataset
    from rfdetr.utils.visualize import visualize_dataset

    register_mammo_dataset(train_json=..., val_json=..., images_dir=...)
    visualize_dataset("mammo_train", num_images=3, seed=42)
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

from rfdetr.data.registration import register_mammo_dataset
from rfdetr.utils.visualize import visualize_dataset


def main():
    parser = argparse.ArgumentParser(description="Visualize mammography dataset")
    parser.add_argument("--train-json",  required=True)
    parser.add_argument("--val-json",    required=True)
    parser.add_argument("--images-dir",  required=True)
    parser.add_argument("--split",       choices=["train", "val"], default="train")
    parser.add_argument("--num-images",  type=int, default=3)
    parser.add_argument("--low-pct",     type=float, default=1.0)
    parser.add_argument("--high-pct",    type=float, default=99.0)
    parser.add_argument("--save-dir",    default=None, help="Save figures here instead of showing")
    parser.add_argument("--seed",        type=int, default=42)
    args = parser.parse_args()

    register_mammo_dataset(
        train_json=args.train_json,
        val_json=args.val_json,
        images_dir=args.images_dir,
    )

    dataset_name = f"mammo_{args.split}"
    visualize_dataset(
        dataset_name=dataset_name,
        num_images=args.num_images,
        low_pct=args.low_pct,
        high_pct=args.high_pct,
        save_dir=args.save_dir,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
