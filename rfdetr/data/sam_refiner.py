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

    Returns:
        Liste de dictionnaires :
        [{
            "box": [x, y, w, h],
            "mask": component_mask,
            "reduction_pct": float,
            "is_split": bool,
            "component_idx": int,
            "total_components": int,
        }, ...]
    """
    orig_x, orig_y, orig_w, orig_h = original_box_xywh
    orig_area = max(1.0, orig_w * orig_h)

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

    # Cas 1 seule composante OU splitting désactivé
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


def refine_box_from_mask(
    mask: np.ndarray,
    original_box_xywh: List[float],
    min_area_ratio: float = 0.10,
    max_area_ratio: float = 1.05,
) -> Tuple[List[float], float]:
    """Extrait la boîte englobante minimale d'un masque binaire (compatibilité descendante)."""
    items = extract_refined_boxes_from_mask(
        mask=mask,
        original_box_xywh=original_box_xywh,
        split_multi=False,
        min_area_ratio=min_area_ratio,
        max_area_ratio=max_area_ratio,
    )
    return items[0]["box"], items[0]["reduction_pct"]


def segment_box_with_sam(
    predictor: Any,
    image_rgb: np.ndarray,
    box_xywh: List[float],
    multimask_output: bool = True,
    set_image: bool = True,
    split_multi_lesions: bool = True,
) -> Tuple[List[Dict[str, Any]], np.ndarray, float, float]:
    """Applique SAM en mode box-prompt pour segmenter la lésion et resserrer la boîte.

    Args:
        predictor: Instance de SamPredictor.
        image_rgb: Image uint8 normalisée (H, W, 3).
        box_xywh: Boîte englobante originale au format COCO [x, y, w, h].
        multimask_output: Si True, évalue les 3 masques candidats de SAM et retient le meilleur.
        set_image: Si True, met à jour l'image interne du predictor.
        split_multi_lesions: Si True, découpe les multi-nodules en sous-boîtes distinctes.

    Returns:
        Tuple : (refined_items_list, best_mask, confidence_score, primary_reduction_pct)
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

    # Contrôle post-traitement : si SAM a segmenté du noir (air ambiant < 20), on garde l'ancienne boîte
    if best_mask is not None and np.any(best_mask):
        gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY) if image_rgb.ndim == 3 else image_rgb
        mean_intensity = float(np.mean(gray[best_mask > 0]))
        if mean_intensity < 20.0:
            logger.info(
                f"Post-traitement : région segmentée dans le fond noir (intensité {mean_intensity:.1f} < 20.0). "
                "Conservation de l'ancienne boîte originale."
            )
            fallback_items = [{
                "box": list(box_xywh),
                "mask": None,
                "reduction_pct": 0.0,
                "is_split": False,
                "component_idx": 1,
                "total_components": 1,
            }]
            return fallback_items, None, 0.0, 0.0

    refined_items = extract_refined_boxes_from_mask(
        mask=best_mask,
        original_box_xywh=box_xywh,
        split_multi=split_multi_lesions,
    )
    primary_red = refined_items[0]["reduction_pct"]
    return refined_items, best_mask, best_score, primary_red


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
            is_black_fb = (sc == 0.0 and red_pct == 0.0)
            box_col = "#FF9800" if is_black_fb else "#00E676"
            tight_rect = patches.Rectangle(
                (tbx, tby), tbw, tbh,
                linewidth=2.5,
                edgecolor=box_col,
                facecolor="none",
            )
            axs[i, 1].add_patch(tight_rect)

            label_text = f"{cat_name} (Conservé : fond noir)" if is_black_fb else f"SAM : {cat_name} (-{red_pct:.1f}%) [score {sc:.2f}]"
            lbl_bg = "#FF9800" if is_black_fb else "#00E676"
            axs[i, 1].text(
                tbx + 4, max(14, tby - 6),
                label_text,
                color="black",
                fontsize=11,
                fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.2", facecolor=lbl_bg, alpha=0.9, edgecolor="none"),
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
    from rfdetr.utils.visualize import read_mammo_uint8, resolve_image_path

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

    # Recherche multi-dossiers intelligente (dossier spécifié + dossiers usuels Colab et Drive)
    search_dirs = [images_dir] if images_dir else []
    for cand in [
        "/content/mammo_data/images",
        "/content/mammo_data",
        "/content/images",
        "/content/drive/MyDrive/EMBED_Dataset/images",
        "/content/drive/MyDrive/EMBED_Dataset/curated/full_dataset/images",
        "/content/drive/MyDrive/EMBED_Dataset",
    ]:
        if os.path.isdir(cand) and cand not in search_dirs:
            search_dirs.append(cand)

    valid_images_with_lesions = []
    effective_dir = images_dir
    for s_dir in search_dirs:
        for img in coco_data.get("images", []):
            if len(img_to_annos.get(img["id"], [])) > 0:
                resolved = resolve_image_path(img["file_name"], images_fallback_dir=s_dir)
                if resolved:
                    valid_images_with_lesions.append((img, resolved))
        if valid_images_with_lesions:
            effective_dir = s_dir
            if s_dir != images_dir:
                print(f"💡 Clichés localisés automatiquement dans : {effective_dir}")
            break

    if not valid_images_with_lesions:
        sample_expected = coco_data["images"][0]["file_name"] if coco_data.get("images") else "image.png"
        checked_list = "\n   - ".join(search_dirs) if search_dirs else "aucun dossier valide"
        raise FileNotFoundError(
            f"Aucun fichier image correspondant au JSON n'a été trouvé.\n"
            f"Fichier recherché (ex) : {sample_expected}\n"
            f"Dossiers inspectés :\n   - {checked_list}\n\n"
            f"💡 Dans Google Colab, pour localiser l'emplacement réel de ce fichier, lancez :\n"
            f"   !find /content -name \"*{Path(sample_expected).name}*\"\n"
        )

    print(f"   Clichés annotés identifiés sur le disque : {len(valid_images_with_lesions)}")

    random.seed(seed)
    selected_pairs = random.sample(valid_images_with_lesions, min(num_samples, len(valid_images_with_lesions)))

    # Chargement de SAM
    predictor = load_sam_predictor(model_type=model_type, checkpoint_path=checkpoint_path)

    results_list = []
    summary_table = []

    print(f"\nTraitement des {len(selected_pairs)} clichés par SAM en cours...")
    for idx, (img_info, resolved_path) in enumerate(selected_pairs, start=1):
        file_name = img_info["file_name"]
        annos = img_to_annos[img_info["id"]]

        try:
            # Lecture de la mammographie 16-bit normalisée en RGB uint8
            img_rgb = read_mammo_uint8(resolved_path)

            # Encode l'image une seule fois dans le ViT de SAM
            predictor.set_image(img_rgb)
        except Exception as e:
            logger.warning(f"Impossible de traiter {file_name}: {e}. Cliché ignoré.")
            continue

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

            refined_items, mask, score, _ = segment_box_with_sam(
                predictor=predictor,
                image_rgb=img_rgb,
                box_xywh=orig_box,
                set_image=False,  # Image déjà encodée
                split_multi_lesions=split_multi_lesions,
            )

            for item in refined_items:
                tight_box = item["box"]
                red_pct = item["reduction_pct"]
                is_split = item.get("is_split", False)
                comp_idx = item.get("component_idx", 1)
                total_comps = item.get("total_components", 1)

                display_cat = f"{cat_name} #{comp_idx}" if is_split else cat_name
                sample_res["orig_boxes"].append(orig_box)
                sample_res["tight_boxes"].append(tight_box)
                sample_res["masks"].append(item.get("mask", mask))
                sample_res["reductions"].append(red_pct)
                sample_res["scores"].append(score)
                sample_res["cat_names"].append(display_cat)

                orig_w, orig_h = orig_box[2], orig_box[3]
                new_w, new_h = tight_box[2], tight_box[3]
                split_tag = f" (Nodule {comp_idx}/{total_comps})" if is_split else ""
                summary_table.append({
                    "index": idx,
                    "file": Path(file_name).name[:26] + "...",
                    "class": f"{cat_name}{split_tag}",
                    "orig_box": f"{int(orig_w)}×{int(orig_h)} px",
                    "tight_box": f"{int(new_w)}×{int(new_h)} px",
                    "reduction": f"-{red_pct:.1f}%",
                    "sam_score": f"{score:.2f}",
                })

        results_list.append(sample_res)
        n_res_boxes = len(sample_res["tight_boxes"])
        split_note = f" (scission en {n_res_boxes} nodules)" if n_res_boxes > len(annos) else ""
        print(f"   [{idx:02d}/{len(selected_pairs):02d}] ✅ {Path(file_name).name} traité ({n_res_boxes} boîte(s){split_note})")

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
    if save_dir is None or save_dir == "./sam_preview":
        save_dir = "/content" if os.path.exists("/content") else "./sam_preview"
    os.makedirs(save_dir, exist_ok=True)
    out_img_path = os.path.join(save_dir, "sam_refinement_preview_10_samples.png")
    visualize_sam_comparison_grid(
        samples_results=results_list,
        save_path=out_img_path,
        show=show,
    )

    return {
        "num_samples": len(selected_pairs),
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
            orig_box = ann["bbox"]
            refined_items, _, score, _ = segment_box_with_sam(
                predictor=predictor,
                image_rgb=img_rgb,
                box_xywh=orig_box,
                set_image=False,
                split_multi_lesions=split_multi_lesions,
            )
            for k, item in enumerate(refined_items):
                tight_box = item["box"]
                red_pct = item["reduction_pct"]
                new_ann = dict(ann)
                if len(refined_items) > 1:
                    new_ann["id"] = int(f"{ann['id']}{k+1}")
                    new_ann["split_from_id"] = ann["id"]
                    new_ann["component_index"] = k + 1
                    new_ann["total_components"] = len(refined_items)

                new_ann["bbox"] = [round(v, 2) for v in tight_box]
                new_ann["area"] = round(tight_box[2] * tight_box[3], 2)
                new_ann["sam_refinement"] = {
                    "original_bbox": orig_box,
                    "reduction_pct": round(red_pct, 2),
                    "sam_score": round(score, 3),
                    "is_split": item.get("is_split", False),
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
