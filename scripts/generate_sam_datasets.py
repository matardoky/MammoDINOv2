#!/usr/bin/env python
"""scripts/generate_sam_datasets.py

Script automatisé et autonome pour générer les datasets COCO raffinés par SAM :
  - mass_train.json -> mass_train_sam.json
  - mass_val.json   -> mass_val_sam.json

Mode : Simple Redimensionnement (Normalisation pure sans splitting).
Chaque boîte d'origine est resserrée au plus près des contours de la tumeur via SAM.
Aucune dépendance à Detectron2 ou Detrex. Fonctionne sur n'importe quel GPU (Colab T4).

Exemple d'utilisation :
    python scripts/generate_sam_datasets.py \
        --train-json /content/drive/MyDrive/EMBED_Dataset/curated/full_dataset/coco/mass_train.json \
        --val-json   /content/drive/MyDrive/EMBED_Dataset/curated/full_dataset/coco/mass_val.json \
        --images-dir /content/drive/MyDrive/EMBED_Dataset/curated/full_dataset/images
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s: %(message)s")
logger = logging.getLogger("sam_dataset_generator")

SAM_CHECKPOINTS = {
    "vit_b": {
        "url": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth",
        "filename": "sam_vit_b_01ec64.pth",
        "size_mb": 357.7,
    },
}


def download_sam_weights(target_dir: str = "/content") -> str:
    """Télécharge automatiquement les poids SAM vit_b s'ils ne sont pas déjà présents."""
    info = SAM_CHECKPOINTS["vit_b"]
    os.makedirs(target_dir, exist_ok=True)
    dst_path = os.path.join(target_dir, info["filename"])

    if os.path.isfile(dst_path) and os.path.getsize(dst_path) > 100 * 1024 * 1024:
        logger.info(f"Poids SAM vit_b trouvés : {dst_path}")
        return dst_path

    logger.info(f"📥 Téléchargement de SAM vit_b depuis Meta AI ({info['size_mb']} MB)...")

    def _progress(count, block_size, total_size):
        done_mb = (count * block_size) / (1024 * 1024)
        tot_mb = total_size / (1024 * 1024)
        pct = min(100.0, (done_mb / max(0.1, tot_mb)) * 100.0)
        sys.stdout.write(f"\r   Progression : {pct:.1f}% ({done_mb:.1f}/{tot_mb:.1f} MB)")
        sys.stdout.flush()

    urllib.request.urlretrieve(info["url"], dst_path, reporthook=_progress)
    print("\n✅ Poids SAM téléchargés !")
    return dst_path


def read_mammo_normalized_rgb(file_path: str, low_pct: float = 1.0, high_pct: float = 99.0) -> np.ndarray:
    """Lit une mammographie 16-bit ou 8-bit et produit une image RGB uint8 normalisée."""
    img = cv2.imread(file_path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise IOError(f"Impossible de lire l'image : {file_path}")

    if img.ndim == 3:
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    if img.dtype == np.uint16:
        fg_mask = img > np.percentile(img, 5.0)
        fg_pixels = img[fg_mask] if np.count_nonzero(fg_mask) > 100 else img
        lo = float(np.percentile(fg_pixels, low_pct))
        hi = float(np.percentile(fg_pixels, high_pct))
    else:
        lo = float(np.percentile(img, low_pct))
        hi = float(np.percentile(img, high_pct))

    if hi <= lo:
        hi = lo + 1.0

    norm = np.clip((img.astype(np.float32) - lo) / (hi - lo), 0.0, 1.0)
    u8 = (norm * 255.0).astype(np.uint8)
    return np.stack([u8, u8, u8], axis=-1)


def refine_box_tight_only(
    mask: np.ndarray,
    original_box_xywh: List[float],
    min_area_ratio: float = 0.06,
    max_area_ratio: float = 1.05,
) -> Tuple[List[float], float]:
    """Resserre la boîte au plus près du masque sans découpage (préservation 1-pour-1)."""
    orig_x, orig_y, orig_w, orig_h = original_box_xywh
    orig_area = max(1.0, orig_w * orig_h)

    if mask is None or not np.any(mask):
        return list(original_box_xywh), 0.0

    ys, xs = np.where(mask > 0)
    if len(xs) == 0 or len(ys) == 0:
        return list(original_box_xywh), 0.0

    x_min, x_max = float(np.min(xs)), float(np.max(xs))
    y_min, y_max = float(np.min(ys)), float(np.max(ys))

    new_w = max(1.0, x_max - x_min + 1.0)
    new_h = max(1.0, y_max - y_min + 1.0)
    new_area = new_w * new_h
    ratio = new_area / orig_area

    # Garde-fous de sécurité
    if ratio < min_area_ratio or ratio > max_area_ratio:
        return list(original_box_xywh), 0.0

    red_pct = (1.0 - (new_area / orig_area)) * 100.0
    return [round(x_min, 2), round(y_min, 2), round(new_w, 2), round(new_h, 2)], round(red_pct, 2)


def process_single_coco_json(
    input_json_path: str,
    output_json_path: str,
    images_dir: str,
    predictor: Any,
    file_index: Dict[str, str],
) -> Dict[str, Any]:
    """Traite un fichier JSON COCO complet et exporte la version resserrée par SAM."""
    print("=" * 80)
    print(f"📄 Traitement du dataset : {Path(input_json_path).name}")
    print(f"   Entrée  : {input_json_path}")
    print(f"   Sortie  : {output_json_path}")
    print("=" * 80)

    with open(input_json_path, "r", encoding="utf-8") as f:
        coco_data = json.load(f)

    # Indexer annotations par image_id
    img_to_annos: Dict[int, List[Dict[str, Any]]] = {}
    for ann in coco_data.get("annotations", []):
        img_to_annos.setdefault(ann["image_id"], []).append(ann)

    id_to_img = {img["id"]: img for img in coco_data.get("images", [])}

    refined_annotations = []
    total_reductions = []
    skipped_count = 0
    start_time = time.time()

    images_to_process = list(img_to_annos.items())
    total_imgs = len(images_to_process)

    for idx, (img_id, annos) in enumerate(images_to_process, start=1):
        img_info = id_to_img.get(img_id)
        if not img_info:
            continue

        fname = img_info["file_name"]
        base = os.path.basename(fname)

        # Résolution du chemin image
        file_path = None
        direct = os.path.join(images_dir, fname)
        if os.path.isfile(direct):
            file_path = direct
        elif base in file_index:
            file_path = file_index[base]
        elif base.lower() in file_index:
            file_path = file_index[base.lower()]

        if not file_path:
            # Cliché introuvable -> conservation telle quelle
            refined_annotations.extend(annos)
            skipped_count += len(annos)
            continue

        try:
            img_rgb = read_mammo_normalized_rgb(file_path)
            predictor.set_image(img_rgb)
        except Exception as e:
            logger.warning(f"Erreur de lecture sur {fname}: {e}. Boîtes conservées.")
            refined_annotations.extend(annos)
            skipped_count += len(annos)
            continue

        for ann in annos:
            orig_box = ann["bbox"]
            bx, by, bw, bh = orig_box
            box_prompt = np.array([bx, by, bx + bw, by + bh], dtype=np.float32)

            masks, scores, _ = predictor.predict(
                box=box_prompt[None, :],
                multimask_output=True,
            )

            best_idx = int(np.argmax(scores))
            best_mask = masks[best_idx]
            best_score = float(scores[best_idx])

            tight_box, red_pct = refine_box_tight_only(best_mask, orig_box)

            new_ann = dict(ann)
            new_ann["bbox"] = tight_box
            new_ann["area"] = round(tight_box[2] * tight_box[3], 2)
            new_ann["sam_refinement"] = {
                "original_bbox": orig_box,
                "reduction_pct": red_pct,
                "sam_score": round(best_score, 3),
            }
            refined_annotations.append(new_ann)
            total_reductions.append(red_pct)

        if idx % 25 == 0 or idx == total_imgs:
            elapsed = time.time() - start_time
            speed = idx / max(0.1, elapsed)
            remain_sec = (total_imgs - idx) / max(0.01, speed)
            sys.stdout.write(
                f"\r   [{idx}/{total_imgs}] {pct_done:.1f}% | "
                f"{speed:.1f} img/s | Restant : {remain_sec:.0f}s | "
                f"Lésions : {len(refined_annotations)} | Réd. moy. : -{np.mean(total_reductions or [0]):.1f}%"
                .format(pct_done=(idx / total_imgs) * 100)
            )
            sys.stdout.flush()

    # Sauvegarde du nouveau JSON COCO
    new_coco_data = dict(coco_data)
    new_coco_data["annotations"] = refined_annotations

    os.makedirs(os.path.dirname(os.path.abspath(output_json_path)), exist_ok=True)
    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump(new_coco_data, f, indent=2)

    avg_red = float(np.mean(total_reductions)) if total_reductions else 0.0
    print(f"\n✅ Fichier sauvegardé : {output_json_path}")
    print(f"   Total annotations : {len(refined_annotations)}")
    print(f"   Réduction moyenne : -{avg_red:.1f}%")
    if skipped_count > 0:
        print(f"   ⚠️ Annotations conservées intactes (images non trouvées) : {skipped_count}")
    print()

    return {
        "output_path": output_json_path,
        "total_annotations": len(refined_annotations),
        "average_reduction": avg_red,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Génération des datasets COCO raffinés par SAM (mass_train_sam.json & mass_val_sam.json)"
    )
    parser.add_argument(
        "--train-json",
        default="/content/drive/MyDrive/EMBED_Dataset/curated/full_dataset/coco/mass_train.json",
        help="Chemin vers mass_train.json",
    )
    parser.add_argument(
        "--val-json",
        default="/content/drive/MyDrive/EMBED_Dataset/curated/full_dataset/coco/mass_val.json",
        help="Chemin vers mass_val.json",
    )
    parser.add_argument(
        "--images-dir",
        default="/content/drive/MyDrive/EMBED_Dataset/curated/full_dataset/images",
        help="Dossier contenant les images mammographiques",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Dossier de sortie (par défaut : même dossier que les JSON d'entrée)",
    )
    args = parser.parse_args()

    # Vérification des fichiers d'entrée
    for p, label in [(args.train_json, "train-json"), (args.val_json, "val-json")]:
        if not os.path.isfile(p):
            raise FileNotFoundError(f"Fichier {label} introuvable : {p}")

    if not os.path.isdir(args.images_dir):
        raise FileNotFoundError(f"Dossier images introuvable : {args.images_dir}")

    # Détermination des chemins de sortie (_sam.json)
    out_dir = args.output_dir or os.path.dirname(os.path.abspath(args.train_json))
    out_train_json = os.path.join(out_dir, "mass_train_sam.json")
    out_val_json = os.path.join(out_dir, "mass_val_sam.json")

    # Indexation récursive des fichiers images pour accès O(1)
    print("🔍 Indexation des images dans :", args.images_dir)
    file_index = {}
    for root, _, files in os.walk(args.images_dir):
        for f in files:
            full_p = os.path.join(root, f)
            file_index[f] = full_p
            file_index[f.lower()] = full_p
    print(f"✅ {len(file_index)} fichiers images indexés.\n")

    # Chargement du modèle SAM
    try:
        from segment_anything import SamPredictor, sam_model_registry
    except ImportError:
        raise ImportError(
            "Le package 'segment-anything' est requis.\n"
            "Installez-le avec : pip install git+https://github.com/facebookresearch/segment-anything.git"
        )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"⚙️ Initialisation de SAM (vit_b) sur {device.upper()}...")
    weights_path = download_sam_weights(target_dir="/content")
    sam_model = sam_model_registry["vit_b"](checkpoint=weights_path)
    sam_model.to(device=device)
    sam_model.eval()
    predictor = SamPredictor(sam_model)
    print("✅ Modèle SAM chargé et prêt.\n")

    # 1. Traitement du dataset de validation
    val_res = process_single_coco_json(
        input_json_path=args.val_json,
        output_json_path=out_val_json,
        images_dir=args.images_dir,
        predictor=predictor,
        file_index=file_index,
    )

    # 2. Traitement du dataset d'entraînement
    train_res = process_single_coco_json(
        input_json_path=args.train_json,
        output_json_path=out_train_json,
        images_dir=args.images_dir,
        predictor=predictor,
        file_index=file_index,
    )

    print("=" * 80)
    print("🎉 GÉNÉRATION DES DATASETS SAM TERMINÉE AVEC SUCCÈS !")
    print(f"   Validation   : {val_res['output_path']} (Réduction : -{val_res['average_reduction']:.1f}%)")
    print(f"   Entraînement : {train_res['output_path']} (Réduction : -{train_res['average_reduction']:.1f}%)")
    print("=" * 80)
    print("\n💡 Vous pouvez maintenant affiner votre modèle avec ces deux nouveaux fichiers !")


if __name__ == "__main__":
    main()
