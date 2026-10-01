"""rfdetr.data.sam_refiner

Raffine les annotations de boîtes englobantes mammographiques en utilisant
Segment Anything Model (SAM) en mode Box-Prompt.

Fonctionnalités :
  1. Chargement et téléchargement automatique de SAM (ViT-B / ViT-L).
  2. Segmentation intra-boîte guidée par la vérité terrain humaine (box prompt).
  3. Extraction automatique du rectangle englobant minimal (tight bbox) à partir du masque.
  4. Rendu comparatif en 2 colonnes (Avant / Après SAM) pour validation visuelle sur 10 clichés.
  5. Exportation vers un nouveau fichier JSON COCO prêt pour l'entraînement.
"""

from __future__ import annotations

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

logger = logging.getLogger("sam_refiner")

SAM_CHECKPOINT_URLS = {
    "vit_b": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth",
    "vit_l": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth",
    "vit_h": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth",
}


def download_sam_checkpoint(model_type: str = "vit_b", target_dir: str = "./weights") -> str:
    """Télécharge le checkpoint officiel de SAM s'il n'est pas déjà présent localement.

    Args:
        model_type: Type d'architecture SAM ('vit_b', 'vit_l', 'vit_h').
        target_dir: Dossier de destination pour stocker les poids.

    Returns:
        Chemin absolu vers le fichier de poids .pth.
    """
    if model_type not in SAM_CHECKPOINT_URLS:
        raise ValueError(f"Modèle SAM inconnu : {model_type}. Choisir parmi : {list(SAM_CHECKPOINT_URLS.keys())}")

    os.makedirs(target_dir, exist_ok=True)
    url = SAM_CHECKPOINT_URLS[model_type]
    filename = os.path.basename(url)
    dest_path = os.path.join(target_dir, filename)

    if os.path.isfile(dest_path) and os.path.getsize(dest_path) > 10_000_000:
        logger.info(f"Poids SAM trouvés : {dest_path}")
        return dest_path

    print(f"📥 Téléchargement des poids SAM ({model_type}) depuis Meta AI...")
    print(f"   URL : {url}")
    print(f"   Destination : {dest_path}")

    def _progress_hook(count, block_size, total_size):
        if total_size > 0:
            percent = int(count * block_size * 100 / total_size)
            mb_downloaded = (count * block_size) / (1024 * 1024)
            mb_total = total_size / (1024 * 1024)
            sys.stdout.write(f"\r   Progression : {percent:3d}% ({mb_downloaded:.1f}/{mb_total:.1f} MB)")
            sys.stdout.flush()

    urllib.request.urlretrieve(url, dest_path, reporthook=_progress_hook)
    print("\n✅ Téléchargement terminé avec succès !")
    return dest_path


def load_sam_predictor(
    model_type: str = "vit_b",
    checkpoint_path: Optional[str] = None,
    device: Optional[str] = None,
) -> Any:
    """Instancie le modèle SAM et retourne un objet SamPredictor prêt à l'emploi.

    Args:
        model_type: Architecture SAM ('vit_b', 'vit_l', 'vit_h').
        checkpoint_path: Chemin vers le checkpoint .pth (téléchargé si None).
        device: 'cuda' ou 'cpu' (détecté automatiquement si None).

    Returns:
        SamPredictor configuré.
    """
    try:
        import torch
        from segment_anything import SamPredictor, sam_model_registry
    except ImportError:
        raise ImportError(
            "Le package 'segment-anything' est requis pour ce module.\n"
            "Installez-le avec : pip install segment-anything"
        )

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    if checkpoint_path is None or not os.path.isfile(checkpoint_path):
        checkpoint_path = download_sam_checkpoint(model_type=model_type)

    logger.info(f"Chargement de SAM ({model_type}) sur {device} depuis {checkpoint_path}...")
    sam = sam_model_registry[model_type](checkpoint=checkpoint_path)
    sam.to(device=device)
    sam.eval()
    predictor = SamPredictor(sam)
    logger.info("SAM initialisé avec succès.")
    return predictor


def refine_box_from_mask(
    mask: np.ndarray,
    original_box_xywh: List[float],
    min_area_ratio: float = 0.10,
    max_area_ratio: float = 1.05,
) -> Tuple[List[float], float]:
    """Extrait la boîte englobante minimale d'un masque binaire avec garde-fous de sécurité.

    Args:
        mask: Masque binaire 2D (H, W) de booléens ou entiers.
        original_box_xywh: Boîte originale [x, y, w, h].
        min_area_ratio: Seuil minimal d'aire (si la boîte resserrée < 10% de l'originale, fallback).
        max_area_ratio: Seuil maximal d'aire (si la boîte resserrée > 105% de l'originale, fallback).

    Returns:
        Tuple: (tight_box_xywh, area_reduction_pct)
    """
    orig_x, orig_y, orig_w, orig_h = original_box_xywh
    orig_area = max(1.0, orig_w * orig_h)

    y_indices, x_indices = np.where(mask > 0)
    if len(x_indices) == 0 or len(y_indices) == 0:
        # Masque vide : fallback sécurisé sur la boîte d'origine
        return list(original_box_xywh), 0.0

    x_min, x_max = float(np.min(x_indices)), float(np.max(x_indices))
    y_min, y_max = float(np.min(y_indices)), float(np.max(y_indices))

    new_w = max(1.0, x_max - x_min + 1.0)
    new_h = max(1.0, y_max - y_min + 1.0)
    new_area = new_w * new_h
    ratio = new_area / orig_area

    # Garde-fous : si le masque est trop petit (bruit) ou délirant
    if ratio < min_area_ratio or ratio > max_area_ratio:
        return list(original_box_xywh), 0.0

    tight_box = [x_min, y_min, new_w, new_h]
    reduction_pct = (1.0 - (new_area / orig_area)) * 100.0
    return tight_box, reduction_pct


def segment_box_with_sam(
    predictor: Any,
    image_rgb: np.ndarray,
    box_xywh: List[float],
    multimask_output: bool = True,
    set_image: bool = True,
) -> Tuple[List[float], np.ndarray, float, float]:
    """Applique SAM en mode box-prompt pour segmenter la lésion et resserrer la boîte.

    Args:
        predictor: Instance de SamPredictor.
        image_rgb: Image uint8 normalisée (H, W, 3).
        box_xywh: Boîte englobante originale au format COCO [x, y, w, h].
        multimask_output: Si True, évalue les 3 masques candidats de SAM et retient le meilleur.
        set_image: Si True, met à jour l'image interne du predictor.

    Returns:
        Tuple : (tight_box_xywh, best_mask, confidence_score, reduction_pct)
    """
    if set_image:
        predictor.set_image(image_rgb)

    x, y, w, h = box_xywh
    input_box = np.array([x, y, x + w, y + h])

    masks, scores, _ = predictor.predict(
        point_coords=None,
        point_labels=None,
        box=input_box[None, :],
        multimask_output=multimask_output,
    )

    if multimask_output:
        # Sélection du masque ayant le score de prédiction IoU le plus élevé
        best_idx = int(np.argmax(scores))
        best_mask = masks[best_idx]
        best_score = float(scores[best_idx])
    else:
        best_mask = masks[0]
        best_score = float(scores[0]) if len(scores) > 0 else 1.0

    tight_box, reduction_pct = refine_box_from_mask(best_mask, box_xywh)
    return tight_box, best_mask, best_score, reduction_pct


def visualize_sam_comparison_grid(
    samples_results: List[Dict[str, Any]],
    save_path: Optional[str] = None,
    show: bool = False,
) -> Optional[str]:
    """Dessine une figure comparative en 2 colonnes pour les 10 exemples testés :
      - Colonne 1 (Gauche) : Image avec la boîte originale du dataset (Ground Truth).
      - Colonne 2 (Droite) : Image avec le masque de segmentation SAM + boîte resserrée (verte) + boîte originale en pointillés (rouge).

    Args:
        samples_results: Liste de dictionnaires contenant pour chaque échantillon :
          'image_rgb', 'file_name', 'orig_boxes', 'tight_boxes', 'masks', 'reductions', 'scores'.
        save_path: Chemin du fichier PNG de sortie.
        show: Afficher la figure inline avec plt.show().

    Returns:
        Chemin du fichier sauvegardé.
    """
    try:
        import matplotlib
        if not show and not os.environ.get("DISPLAY"):
            matplotlib.use("Agg")
        import matplotlib.patches as patches
        import matplotlib.pyplot as plt
    except ImportError:
        raise ImportError("matplotlib est requis pour la visualisation : pip install matplotlib")

    n_samples = len(samples_results)
    if n_samples == 0:
        logger.warning("Aucun échantillon à visualiser.")
        return None

    fig, axs = plt.subplots(
        nrows=n_samples,
        ncols=2,
        figsize=(16, 7 * n_samples),
        squeeze=False,
    )

    for i, res in enumerate(samples_results):
        img_rgb = res["image_rgb"]
        file_name = Path(res["file_name"]).name
        orig_boxes = res["orig_boxes"]
        tight_boxes = res["tight_boxes"]
        masks = res["masks"]
        reductions = res["reductions"]
        scores = res["scores"]
        cat_names = res.get("cat_names", ["Mass"] * len(orig_boxes))

        # ── Colonne 1 : Avant (Ground Truth Dataset) ────────────────────────
        axs[i, 0].imshow(img_rgb)
        axs[i, 0].axis("off")
        for box, cat_name in zip(orig_boxes, cat_names):
            bx, by, bw, bh = box
            rect = patches.Rectangle(
                (bx, by), bw, bh,
                linewidth=2.5,
                edgecolor="#E53935",  # Rouge vif
                facecolor="none",
            )
            axs[i, 0].add_patch(rect)
            axs[i, 0].text(
                bx + 4, max(14, by - 6),
                f"{cat_name} (Large)",
                color="white",
                fontsize=11,
                fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.2", facecolor="#E53935", alpha=0.85, edgecolor="none"),
            )

        n_lesions = len(orig_boxes)
        axs[i, 0].set_title(
            f"COLONNE 1 : Avant — Boîte Ground Truth Dataset\n"
            f"Fichier : {file_name} ({n_lesions} lésion{'s' if n_lesions > 1 else ''})",
            fontsize=12,
            fontweight="bold",
            color="navy",
        )

        # ── Colonne 2 : Après (Segmentation SAM & Boîte Resserrée) ───────────
        overlay = img_rgb.copy()
        for mask in masks:
            if mask is not None and np.any(mask):
                # Teinte cyan / turquoise translucide pour le masque SAM
                color_mask = np.array([0, 220, 255], dtype=np.uint8)
                overlay[mask > 0] = (overlay[mask > 0] * 0.4 + color_mask * 0.6).astype(np.uint8)

        axs[i, 1].imshow(overlay)
        axs[i, 1].axis("off")

        total_reduction = 0.0
        for orig_b, tight_b, red_pct, sc, cat_name in zip(orig_boxes, tight_boxes, reductions, scores, cat_names):
            # Boîte originale en pointillés rouges (pour repère)
            obx, oby, obw, obh = orig_b
            orig_rect = patches.Rectangle(
                (obx, oby), obw, obh,
                linewidth=1.8,
                edgecolor="#E53935",
                facecolor="none",
                linestyle="--",
            )
            axs[i, 1].add_patch(orig_rect)

            # Nouvelle boîte resserrée en vert néon
            tbx, tby, tbw, tbh = tight_b
            tight_rect = patches.Rectangle(
                (tbx, tby), tbw, tbh,
                linewidth=2.5,
                edgecolor="#00E676",  # Vert éclatant
                facecolor="none",
            )
            axs[i, 1].add_patch(tight_rect)

            label_text = f"SAM : {cat_name} (-{red_pct:.1f}%) [score {sc:.2f}]"
            axs[i, 1].text(
                tbx + 4, max(14, tby - 6),
                label_text,
                color="black",
                fontsize=11,
                fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.2", facecolor="#00E676", alpha=0.9, edgecolor="none"),
            )
            total_reduction += red_pct

        avg_red = total_reduction / max(1, len(orig_boxes))
        axs[i, 1].set_title(
            f"COLONNE 2 : Après — Segmentation SAM & Boîte Resserrée\n"
            f"Masque Cyan + Boîte Verte (Réduction moyenne de surface : -{avg_red:.1f}%)",
            fontsize=12,
            fontweight="bold",
            color="darkgreen",
        )

    plt.suptitle(
        f"Raffinement des Annotations Mammographiques avec SAM (Test sur {n_samples} Clichés)",
        fontsize=15,
        fontweight="bold",
        y=1.002,
    )
    plt.tight_layout()

    out_file = None
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=140, bbox_inches="tight")
        print(f"\n📊 Figure comparative enregistrée avec succès : {save_path}")
        out_file = str(save_path)

    if show or not save_path:
        plt.show()

    plt.close(fig)
    return out_file


def run_sam_preview_10_samples(
    json_path: str,
    images_dir: str,
    num_samples: int = 10,
    model_type: str = "vit_b",
    checkpoint_path: Optional[str] = None,
    save_dir: str = "./sam_preview",
    show: bool = False,
    seed: int = 42,
) -> Dict[str, Any]:
    """Exécute le test de faisabilité sur 10 images contenant des lésions annotées.

    Affiche la comparaison 2 colonnes (Avant Ground Truth vs Après SAM) et
    affiche un tableau récapitulatif des réductions de surface.

    Args:
        json_path: Chemin vers le JSON COCO d'entrée (train ou val).
        images_dir: Dossier racine des images mammographiques.
        num_samples: Nombre d'images à tester (défaut : 10).
        model_type: Modèle SAM ('vit_b', 'vit_l').
        checkpoint_path: Chemin optionnel vers le fichier .pth de SAM.
        save_dir: Dossier où enregistrer la figure comparative.
        show: Afficher la figure dans le notebook.
        seed: Graine aléatoire pour sélectionner les 10 images de façon reproductible.

    Returns:
        Dictionnaire récapitulatif des résultats.
    """
    from rfdetr.utils.visualize import read_mammo_uint8

    print("=" * 75)
    print(f"🔬 TEST DE RAFFINEMENT DES ANNOTATIONS AVEC SAM (BOX-PROMPT)")
    print(f"   Dataset JSON  : {json_path}")
    print(f"   Dossier Images: {images_dir}")
    print(f"   Échantillons   : {num_samples} clichés")
    print("=" * 75)

    with open(json_path, "r", encoding="utf-8") as f:
        coco_data = json.load(f)

    cat_map = {c["id"]: c["name"] for c in coco_data.get("categories", [])}

    # Indexation des annotations par image_id
    img_to_annos: Dict[int, List[Dict[str, Any]]] = {}
    for ann in coco_data.get("annotations", []):
        img_id = ann["image_id"]
        img_to_annos.setdefault(img_id, []).append(ann)

    # Filtrer les images contenant au moins une lésion annotée
    images_with_lesions = [
        img for img in coco_data.get("images", [])
        if len(img_to_annos.get(img["id"], [])) > 0
    ]

    if not images_with_lesions:
        raise ValueError(f"Aucune image avec annotation trouvée dans {json_path}")

    random.seed(seed)
    selected_images = random.sample(images_with_lesions, min(num_samples, len(images_with_lesions)))

    # Chargement de SAM
    predictor = load_sam_predictor(model_type=model_type, checkpoint_path=checkpoint_path)

    results_list = []
    summary_table = []

    print("\nTraitement des 10 clichés par SAM en cours...")
    for idx, img_info in enumerate(selected_images, start=1):
        file_name = img_info["file_name"]
        annos = img_to_annos[img_info["id"]]

        # Lecture de la mammographie 16-bit normalisée en RGB uint8
        img_rgb = read_mammo_uint8(file_name, images_fallback_dir=images_dir)

        # Encode l'image une seule fois dans le ViT de SAM
        predictor.set_image(img_rgb)

        sample_res = {
            "image_rgb": img_rgb,
            "file_name": file_name,
            "orig_boxes": [],
            "tight_boxes": [],
            "masks": [],
            "reductions": [],
            "scores": [],
            "cat_names": [],
        }

        for ann in annos:
            orig_box = ann["bbox"]  # [x, y, w, h]
            cat_name = cat_map.get(ann.get("category_id", 1), "Mass")

            tight_box, mask, score, red_pct = segment_box_with_sam(
                predictor=predictor,
                image_rgb=img_rgb,
                box_xywh=orig_box,
                set_image=False,  # Image déjà encodée
            )

            sample_res["orig_boxes"].append(orig_box)
            sample_res["tight_boxes"].append(tight_box)
            sample_res["masks"].append(mask)
            sample_res["reductions"].append(red_pct)
            sample_res["scores"].append(score)
            sample_res["cat_names"].append(cat_name)

            orig_w, orig_h = orig_box[2], orig_box[3]
            new_w, new_h = tight_box[2], tight_box[3]
            summary_table.append({
                "index": idx,
                "file": Path(file_name).name[:28] + "...",
                "class": cat_name,
                "orig_box": f"{int(orig_w)}×{int(orig_h)} px",
                "tight_box": f"{int(new_w)}×{int(new_h)} px",
                "reduction": f"-{red_pct:.1f}%",
                "sam_score": f"{score:.2f}",
            })

        results_list.append(sample_res)
        print(f"   [{idx:02d}/{len(selected_images):02d}] ✅ {Path(file_name).name} traité ({len(annos)} lésion(s))")

    # Affichage du tableau synthétique
    print("\n" + "=" * 90)
    print(f"{'#':<3} | {'Image':<32} | {'Classe':<12} | {'Taille Orig.':<14} | {'Taille SAM':<14} | {'Réduction':<10} | {'Score'}")
    print("-" * 90)
    total_red = 0.0
    for row in summary_table:
        print(f"{row['index']:<3} | {row['file']:<32} | {row['class']:<12} | {row['orig_box']:<14} | {row['tight_box']:<14} | {row['reduction']:<10} | {row['sam_score']}")
        total_red += float(row["reduction"].replace("-", "").replace("%", ""))
    avg_red = total_red / max(1, len(summary_table))
    print("=" * 90)
    print(f"🎯 Réduction moyenne de la surface des boîtes : -{avg_red:.1f}% plus ajustée sur la lésion !")
    print("=" * 90)

    # Sauvegarde et rendu de la figure 2 colonnes
    os.makedirs(save_dir, exist_ok=True)
    out_img_path = os.path.join(save_dir, "sam_refinement_preview_10_samples.png")
    visualize_sam_comparison_grid(
        samples_results=results_list,
        save_path=out_img_path,
        show=show,
    )

    return {
        "num_samples": len(selected_images),
        "total_lesions": len(summary_table),
        "average_reduction_pct": avg_red,
        "preview_image": out_img_path,
        "summary": summary_table,
    }


def refine_entire_coco_dataset(
    input_json_path: str,
    output_json_path: str,
    images_dir: str,
    model_type: str = "vit_b",
    checkpoint_path: Optional[str] = None,
) -> Dict[str, Any]:
    """Applique le raffinement SAM sur l'INTEGRALITE d'un dataset COCO et sauvegarde le nouveau JSON.

    Args:
        input_json_path: Chemin du fichier JSON COCO d'origine.
        output_json_path: Chemin du nouveau fichier JSON COCO resserré.
        images_dir: Dossier des images.
        model_type: Modèle SAM ('vit_b', 'vit_l').
        checkpoint_path: Chemin vers les poids .pth.

    Returns:
        Statistiques globales du traitement.
    """
    from rfdetr.utils.visualize import read_mammo_uint8

    print("=" * 75)
    print("🚀 RAFFINEMENT COMPLET DU DATASET VIA SAM")
    print(f"   Entrée  : {input_json_path}")
    print(f"   Sortie  : {output_json_path}")
    print("=" * 75)

    with open(input_json_path, "r", encoding="utf-8") as f:
        coco_data = json.load(f)

    predictor = load_sam_predictor(model_type=model_type, checkpoint_path=checkpoint_path)

    # Regrouper annotations par image pour encoder chaque image une seule fois
    img_to_annos: Dict[int, List[Dict[str, Any]]] = {}
    for ann in coco_data.get("annotations", []):
        img_to_annos.setdefault(ann["image_id"], []).append(ann)

    id_to_img = {img["id"]: img for img in coco_data.get("images", [])}

    refined_annotations = []
    total_reductions = []
    skipped_count = 0

    total_images = len(img_to_annos)
    for idx, (img_id, annos) in enumerate(img_to_annos.items(), start=1):
        img_info = id_to_img.get(img_id)
        if not img_info:
            continue

        file_name = img_info["file_name"]
        try:
            img_rgb = read_mammo_uint8(file_name, images_fallback_dir=images_dir)
            predictor.set_image(img_rgb)
        except Exception as e:
            logger.warning(f"Impossible de lire l'image {file_name}: {e}. Boîtes conservées telles quelles.")
            refined_annotations.extend(annos)
            skipped_count += len(annos)
            continue

        for ann in annos:
            new_ann = dict(ann)
            orig_box = ann["bbox"]
            tight_box, _, score, red_pct = segment_box_with_sam(
                predictor=predictor,
                image_rgb=img_rgb,
                box_xywh=orig_box,
                set_image=False,
            )
            new_ann["bbox"] = [round(v, 2) for v in tight_box]
            new_ann["area"] = round(tight_box[2] * tight_box[3], 2)
            new_ann["sam_refinement"] = {
                "original_bbox": orig_box,
                "reduction_pct": round(red_pct, 2),
                "sam_score": round(score, 3),
            }
            refined_annotations.append(new_ann)
            total_reductions.append(red_pct)

        if idx % 50 == 0 or idx == total_images:
            sys.stdout.write(f"\r   Images traitées : {idx}/{total_images} ({len(refined_annotations)} lésions)")
            sys.stdout.flush()

    new_coco_data = dict(coco_data)
    new_coco_data["annotations"] = refined_annotations

    os.makedirs(os.path.dirname(os.path.abspath(output_json_path)), exist_ok=True)
    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump(new_coco_data, f, indent=2)

    avg_red = sum(total_reductions) / max(1, len(total_reductions))
    print(f"\n\n✅ Raffinement complet terminé !")
    print(f"   Total lésions raffinées : {len(refined_annotations)}")
    print(f"   Réduction moyenne surface: -{avg_red:.1f}%")
    print(f"   Fichier sauvegardé       : {output_json_path}")
    print("=" * 75)

    return {
        "total_refined": len(refined_annotations),
        "avg_reduction_pct": avg_red,
        "output_json": output_json_path,
    }
