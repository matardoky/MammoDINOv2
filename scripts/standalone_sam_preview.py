#!/usr/bin/env python
"""scripts/standalone_sam_preview.py

Script 100% autonome et indépendant pour tester la segmentation SAM sur 10 clichés mammographiques.
Aucune dépendance à Detectron2 ou Detrex. Nécessite uniquement PyTorch, TorchVision et Segment-Anything.

Utilisation CLI :
    python scripts/standalone_sam_preview.py \
        --json-file /content/drive/MyDrive/EMBED_Dataset/curated/full_dataset/coco/mass_val.json \
        --images-dir /content/drive/MyDrive/EMBED_Dataset/curated/full_dataset/images \
        --num-samples 10 \
        --save-path /content/sam_10_samples_comparison.png
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
import sys
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s: %(message)s")
logger = logging.getLogger("sam_standalone")

SAM_CHECKPOINTS = {
    "vit_b": {
        "url": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth",
        "filename": "sam_vit_b_01ec64.pth",
        "size_mb": 357.7,
    },
    "vit_l": {
        "url": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth",
        "filename": "sam_vit_l_0b3195.pth",
        "size_mb": 1248.6,
    },
}


def download_sam_weights(model_type: str = "vit_b", target_dir: str = "/content") -> str:
    """Télécharge automatiquement les poids officiels de SAM depuis Meta AI."""
    info = SAM_CHECKPOINTS.get(model_type, SAM_CHECKPOINTS["vit_b"])
    os.makedirs(target_dir, exist_ok=True)
    dst_path = os.path.join(target_dir, info["filename"])

    if os.path.isfile(dst_path) and os.path.getsize(dst_path) > 100 * 1024 * 1024:
        logger.info(f"Poids SAM trouvés localement : {dst_path}")
        return dst_path

    logger.info(f"📥 Téléchargement de SAM ({model_type}) depuis Meta AI ({info['size_mb']} MB)...")

    def _progress(count, block_size, total_size):
        done_mb = (count * block_size) / (1024 * 1024)
        tot_mb = total_size / (1024 * 1024)
        pct = min(100.0, (done_mb / max(0.1, tot_mb)) * 100.0)
        sys.stdout.write(f"\r   Progression : {pct:.1f}% ({done_mb:.1f}/{tot_mb:.1f} MB)")
        sys.stdout.flush()

    urllib.request.urlretrieve(info["url"], dst_path, reporthook=_progress)
    print("\n✅ Téléchargement des poids SAM terminé avec succès !")
    return dst_path


def read_mammo_normalized_rgb(file_path: str, low_pct: float = 1.0, high_pct: float = 99.0) -> np.ndarray:
    """Lit une mammographie (8-bit ou 16-bit uint16) et produit un tableau RGB uint8 visible."""
    img = cv2.imread(file_path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise IOError(f"Impossible de lire l'image : {file_path}")

    if img.ndim == 3:
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Grayscale (16-bit ou 8-bit) : normalisation par percentiles avant-plan
    if img.dtype == np.uint16:
        # Masque tissu mammaire (exclure le noir pur du fond)
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


def extract_refined_boxes_from_mask(
    mask: np.ndarray,
    original_box_xywh: List[float],
    split_multi: bool = True,
    min_comp_pixels: int = 64,
    min_comp_ratio: float = 0.12,
    min_area_ratio: float = 0.05,
    max_area_ratio: float = 1.05,
) -> List[Dict[str, Any]]:
    """Extrait une ou plusieurs boîtes englobantes ajustées à partir d'un masque binaire.

    Détecte les composantes connexes via cv2.connectedComponentsWithStats.
    Si split_multi=True et que plusieurs composantes connexes significatives (>12% de la masse)
    sont présentes, découpe la boîte en autant de boîtes distinctes (Option A - Splitting).
    """
    ox, oy, ow, oh = original_box_xywh
    orig_area = max(1.0, ow * oh)

    if mask is None or not np.any(mask):
        return [{
            "box": list(original_box_xywh),
            "mask": mask,
            "reduction_pct": 0.0,
            "is_split": False,
            "component_idx": 1,
            "total_components": 1,
        }]

    u8_mask = (mask > 0).astype(np.uint8)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(u8_mask, connectivity=8)

    total_mask_pixels = float(np.count_nonzero(u8_mask))
    if total_mask_pixels < min_comp_pixels:
        return [{
            "box": list(original_box_xywh),
            "mask": mask,
            "reduction_pct": 0.0,
            "is_split": False,
            "component_idx": 1,
            "total_components": 1,
        }]

    # Identifier les composantes significatives
    valid_components = []
    for k in range(1, num_labels):
        c_area = stats[k, cv2.CC_STAT_AREA]
        if c_area >= min_comp_pixels and (c_area / total_mask_pixels) >= min_comp_ratio:
            valid_components.append(k)

    if len(valid_components) == 0:
        return [{
            "box": list(original_box_xywh),
            "mask": mask,
            "reduction_pct": 0.0,
            "is_split": False,
            "component_idx": 1,
            "total_components": 1,
        }]

    # Cas 1 composante OU splitting désactivé
    if len(valid_components) == 1 or not split_multi:
        if len(valid_components) == 1:
            comp_mask = (labels == valid_components[0])
        else:
            comp_mask = np.isin(labels, valid_components)

        ys, xs = np.where(comp_mask)
        x_min, x_max = float(np.min(xs)), float(np.max(xs))
        y_min, y_max = float(np.min(ys)), float(np.max(ys))
        nw = max(1.0, x_max - x_min + 1.0)
        nh = max(1.0, y_max - y_min + 1.0)
        new_area = nw * nh

        ratio = new_area / orig_area
        if ratio < min_area_ratio or ratio > max_area_ratio:
            return [{
                "box": list(original_box_xywh),
                "mask": mask,
                "reduction_pct": 0.0,
                "is_split": False,
                "component_idx": 1,
                "total_components": 1,
            }]

        red_pct = (1.0 - (new_area / orig_area)) * 100.0
        return [{
            "box": [x_min, y_min, nw, nh],
            "mask": comp_mask,
            "reduction_pct": red_pct,
            "is_split": False,
            "component_idx": 1,
            "total_components": 1,
        }]

    # Cas multi-nodules avec scission (Option A)
    valid_components.sort(key=lambda k: stats[k, cv2.CC_STAT_AREA], reverse=True)
    results = []
    for idx, k in enumerate(valid_components, start=1):
        comp_mask = (labels == k)
        ys, xs = np.where(comp_mask)
        x_min, x_max = float(np.min(xs)), float(np.max(xs))
        y_min, y_max = float(np.min(ys)), float(np.max(ys))
        nw = max(1.0, x_max - x_min + 1.0)
        nh = max(1.0, y_max - y_min + 1.0)
        new_area = nw * nh
        red_pct = (1.0 - (new_area / orig_area)) * 100.0
        results.append({
            "box": [x_min, y_min, nw, nh],
            "mask": comp_mask,
            "reduction_pct": red_pct,
            "is_split": True,
            "component_idx": idx,
            "total_components": len(valid_components),
        })

    return results


def run_standalone_sam_preview(
    json_path: str,
    images_dir: str,
    num_samples: int = 10,
    model_type: str = "vit_b",
    save_path: str = "/content/sam_10_samples_comparison.png",
    seed: int = 42,
    split_multi_lesions: bool = True,
) -> Dict[str, Any]:
    """Exécute la comparaison 2 colonnes Avant/Après SAM sur 10 images sans dépendances tierces."""
    try:
        from segment_anything import SamPredictor, sam_model_registry
    except ImportError:
        raise ImportError(
            "Le package 'segment-anything' est requis.\n"
            "Installez-le avec : pip install git+https://github.com/facebookresearch/segment-anything.git"
        )

    import matplotlib.patches as patches
    import matplotlib.pyplot as plt

    print("=" * 80)
    print("🔬 TEST INDÉPENDANT DU RAFFINEMENT DES BOÎTES AVEC SAM (BOX-PROMPT)")
    print(f"   Annotations JSON : {json_path}")
    print(f"   Dossier Images   : {images_dir}")
    print(f"   Échantillons      : {num_samples} clichés")
    print("=" * 80)

    # 1. Lecture du JSON COCO
    if not os.path.isfile(json_path):
        raise FileNotFoundError(f"Fichier JSON introuvable : {json_path}")

    with open(json_path, "r", encoding="utf-8") as f:
        coco_data = json.load(f)

    cat_map = {c["id"]: c["name"] for c in coco_data.get("categories", [])}

    # Indexer annotations par image_id
    img_to_annos: Dict[int, List[Dict[str, Any]]] = {}
    for ann in coco_data.get("annotations", []):
        img_to_annos.setdefault(ann["image_id"], []).append(ann)

    # 2. Localiser les images existantes sur le disque
    found_pairs: List[Tuple[Dict[str, Any], str]] = []
    print("\n🔍 Vérification des fichiers images sur le disque...")

    # Création d'un index rapide des fichiers présents dans images_dir
    file_index: Dict[str, str] = {}
    if os.path.isdir(images_dir):
        for root, _, files in os.walk(images_dir):
            for f in files:
                file_index[f] = os.path.join(root, f)
                file_index[f.lower()] = os.path.join(root, f)

    for img in coco_data.get("images", []):
        if len(img_to_annos.get(img["id"], [])) == 0:
            continue
        fname = img["file_name"]
        base = os.path.basename(fname)
        resolved = None

        direct = os.path.join(images_dir, fname)
        if os.path.isfile(direct):
            resolved = direct
        elif base in file_index:
            resolved = file_index[base]
        elif base.lower() in file_index:
            resolved = file_index[base.lower()]

        if resolved:
            found_pairs.append((img, resolved))

    if not found_pairs:
        sample_name = coco_data["images"][0]["file_name"] if coco_data.get("images") else "image.png"
        raise FileNotFoundError(
            f"Aucune image du JSON n'a été trouvée dans : {images_dir}\n"
            f"Exemple recherché : {sample_name}\n"
            f"Contenu actuel du dossier (fichiers trouvés) : {len(file_index)} fichiers.\n"
            f"Vérifiez l'emplacement de vos images sur Google Drive."
        )

    print(f"✅ {len(found_pairs)} clichés annotés trouvés avec succès sur le disque !")

    random.seed(seed)
    selected_pairs = random.sample(found_pairs, min(num_samples, len(found_pairs)))

    # 3. Chargement de SAM
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n⚙️ Initialisation de SAM ({model_type}) sur {device.upper()}...")
    weights_path = download_sam_weights(model_type, target_dir="/content")
    sam_model = sam_model_registry[model_type](checkpoint=weights_path)
    sam_model.to(device=device)
    sam_model.eval()
    predictor = SamPredictor(sam_model)
    print("✅ Modèle SAM prêt pour l'inférence !")

    # 4. Traitement des 10 clichés
    samples_results = []
    summary_table = []
    print(f"\nTraitement des {len(selected_pairs)} clichés en cours...")

    for idx, (img_info, file_path) in enumerate(selected_pairs, start=1):
        fname = img_info["file_name"]
        annos = img_to_annos[img_info["id"]]

        try:
            img_rgb = read_mammo_normalized_rgb(file_path)
            predictor.set_image(img_rgb)
        except Exception as e:
            logger.warning(f"Erreur sur {fname}: {e}. Cliché ignoré.")
            continue

        sample_res = {
            "image_rgb": img_rgb,
            "file_name": fname,
            "orig_boxes": [],
            "tight_boxes": [],
            "masks": [],
            "reductions": [],
            "scores": [],
            "cat_names": [],
        }

        for ann in annos:
            orig_b = ann["bbox"]  # [x, y, w, h]
            cat_name = cat_map.get(ann.get("category_id", 1), "Mass")
            bx, by, bw, bh = orig_b

            # Prompt SAM avec la boîte COCO [x1, y1, x2, y2]
            box_prompt = np.array([bx, by, bx + bw, by + bh], dtype=np.float32)
            masks, scores, _ = predictor.predict(
                box=box_prompt[None, :],
                multimask_output=True,
            )

            best_idx = int(np.argmax(scores))
            best_mask = masks[best_idx]
            best_score = float(scores[best_idx])

            refined_items = extract_refined_boxes_from_mask(
                mask=best_mask,
                original_box_xywh=orig_b,
                split_multi=split_multi_lesions,
            )

            for item in refined_items:
                tight_b = item["box"]
                red_pct = item["reduction_pct"]
                is_split = item["is_split"]
                c_idx = item["component_idx"]
                n_comps = item["total_components"]

                disp_cat = f"{cat_name} #{c_idx}" if is_split else cat_name
                sample_res["orig_boxes"].append(orig_b)
                sample_res["tight_boxes"].append(tight_b)
                sample_res["masks"].append(item["mask"])
                sample_res["reductions"].append(red_pct)
                sample_res["scores"].append(best_score)
                sample_res["cat_names"].append(disp_cat)

                summary_table.append({
                    "index": idx,
                    "file": Path(fname).name[:26] + "...",
                    "class": disp_cat,
                    "orig": f"{int(bw)}×{int(bh)}",
                    "tight": f"{int(tight_b[2])}×{int(tight_b[3])}",
                    "reduction": f"-{red_pct:.1f}%",
                    "score": f"{best_score:.2f}",
                })

        samples_results.append(sample_res)
        n_out = len(sample_res["tight_boxes"])
        note = f" (scission en {n_out} nodules)" if n_out > len(annos) else ""
        print(f"   [{idx:02d}/{len(selected_pairs):02d}] ✅ {Path(fname).name} traité ({n_out} boîte(s){note})")

    # 5. Affichage du tableau récapitulatif
    print("\n" + "=" * 85)
    print(f"{'#':<3} | {'Image':<30} | {'Classe':<10} | {'Taille Init.':<12} | {'Taille SAM':<12} | {'Réduction':<10} | {'Score'}")
    print("-" * 85)
    tot_red = 0.0
    for r in summary_table:
        print(f"{r['index']:<3} | {r['file']:<30} | {r['class']:<10} | {r['orig']:<12} | {r['tight']:<12} | {r['reduction']:<10} | {r['score']}")
        tot_red += float(r["reduction"].replace("-", "").replace("%", ""))
    avg_red = tot_red / max(1, len(summary_table))
    print("=" * 85)
    print(f"🎯 RÉDUCTION MOYENNE DE SURFACE : -{avg_red:.1f}% plus ajustée sur la tumeur réelle !")
    print("=" * 85)

    # 6. Rendu de la figure comparative en 2 colonnes
    n = len(samples_results)
    if n > 0:
        fig, axs = plt.subplots(nrows=n, ncols=2, figsize=(16, 7 * n), squeeze=False)
        for i, s in enumerate(samples_results):
            im = s["image_rgb"]
            fn = Path(s["file_name"]).name

            # Colonne 1 : Avant (GT)
            axs[i, 0].imshow(im)
            axs[i, 0].axis("off")
            for b, cname in zip(s["orig_boxes"], s["cat_names"]):
                bx, by, bw, bh = b
                rect = patches.Rectangle((bx, by), bw, bh, linewidth=2.5, edgecolor="#E53935", facecolor="none")
                axs[i, 0].add_patch(rect)
                axs[i, 0].text(
                    bx + 4, max(14, by - 6),
                    f"{cname} (Dataset)",
                    color="white", fontsize=11, fontweight="bold",
                    bbox=dict(boxstyle="round,pad=0.2", facecolor="#E53935", alpha=0.85, edgecolor="none"),
                )
            axs[i, 0].set_title(
                f"COLONNE 1 : Avant — Boîte Ground Truth Dataset\nFichier : {fn}",
                fontsize=12, fontweight="bold", color="navy",
            )

            # Colonne 2 : Après (SAM)
            overlay = im.copy()
            for m in s["masks"]:
                if m is not None and np.any(m):
                    cyan = np.array([0, 220, 255], dtype=np.uint8)
                    overlay[m > 0] = (overlay[m > 0] * 0.4 + cyan * 0.6).astype(np.uint8)

            axs[i, 1].imshow(overlay)
            axs[i, 1].axis("off")
            for ob, tb, red, sc, cname in zip(s["orig_boxes"], s["tight_boxes"], s["reductions"], s["scores"], s["cat_names"]):
                # Repère original
                axs[i, 1].add_patch(patches.Rectangle(
                    (ob[0], ob[1]), ob[2], ob[3],
                    linewidth=1.8, edgecolor="#E53935", facecolor="none", linestyle="--",
                ))
                # Boîte SAM resserrée
                axs[i, 1].add_patch(patches.Rectangle(
                    (tb[0], tb[1]), tb[2], tb[3],
                    linewidth=2.5, edgecolor="#00E676", facecolor="none",
                ))
                axs[i, 1].text(
                    tb[0] + 4, max(14, tb[1] - 6),
                    f"SAM : {cname} (-{red:.1f}%) [score {sc:.2f}]",
                    color="black", fontsize=11, fontweight="bold",
                    bbox=dict(boxstyle="round,pad=0.2", facecolor="#00E676", alpha=0.9, edgecolor="none"),
                )
            axs[i, 1].set_title(
                f"COLONNE 2 : Après — Masque SAM (Cyan) + Boîte Resserrée (Verte)\nFichier : {fn}",
                fontsize=12, fontweight="bold", color="darkgreen",
            )

        plt.suptitle("Comparaison Avant/Après : Resserrement des Boîtes avec SAM (10 Échantillons)", fontsize=15, fontweight="bold", y=1.002)
        plt.tight_layout()

        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        plt.savefig(save_path, dpi=140, bbox_inches="tight")
        print(f"\n📊 Figure comparative enregistrée : {save_path}")
        plt.show()
        plt.close(fig)

    return {
        "num_samples": len(selected_pairs),
        "avg_reduction": avg_red,
        "save_path": save_path,
        "summary": summary_table,
    }


def main():
    parser = argparse.ArgumentParser(description="SAM 10-Samples Standalone Preview")
    parser.add_argument(
        "--json-file",
        default="/content/drive/MyDrive/EMBED_Dataset/curated/full_dataset/coco/mass_val.json",
        help="Chemin vers le fichier JSON COCO",
    )
    parser.add_argument(
        "--images-dir",
        default="/content/drive/MyDrive/EMBED_Dataset/curated/full_dataset/images",
        help="Dossier contenant les images",
    )
    parser.add_argument("--num-samples", type=int, default=10, help="Nombre de clichés (défaut : 10)")
    parser.add_argument("--model-type", choices=["vit_b", "vit_l"], default="vit_b")
    parser.add_argument("--save-path", default="/content/sam_10_samples_comparison.png")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--split-multi-lesions", action="store_true", default=True,
        help="Découper les nodules multiples d'une boîte en sous-boîtes distinctes (Option A)"
    )
    parser.add_argument(
        "--no-split", dest="split_multi_lesions", action="store_false",
        help="Conserver une boîte unique même si plusieurs nodules sont présents (Option B)"
    )
    args = parser.parse_args()

    run_standalone_sam_preview(
        json_path=args.json_file,
        images_dir=args.images_dir,
        num_samples=args.num_samples,
        model_type=args.model_type,
        save_path=args.save_path,
        seed=args.seed,
        split_multi_lesions=args.split_multi_lesions,
    )


if __name__ == "__main__":
    main()
