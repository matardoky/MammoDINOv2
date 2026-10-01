#!/usr/bin/env python
"""scripts/refine_annotations_with_sam.py

Script CLI pour raffiner et resserrer les boîtes englobantes mammographiques avec SAM (Segment Anything Model).

Mode 1 : Test et Visualisation sur 10 clichés (Avant / Après SAM en 2 colonnes) :
    python scripts/refine_annotations_with_sam.py \\
        --json-file  /content/drive/MyDrive/EMBED_Dataset/curated/full_dataset/coco/mass_val.json \\
        --images-dir /content/mammo_data/images \\
        --num-samples 10 \\
        --save-dir   ./sam_preview \\
        --show

Mode 2 : Raffinement complet du dataset entier (génère un nouveau JSON) :
    python scripts/refine_annotations_with_sam.py \\
        --json-file   /content/drive/MyDrive/EMBED_Dataset/curated/full_dataset/coco/mass_train.json \\
        --output-json /content/drive/MyDrive/EMBED_Dataset/curated/full_dataset/coco/mass_train_tight.json \\
        --images-dir  /content/mammo_data/images \\
        --full-dataset
"""

from __future__ import annotations

import argparse
import logging
import os
import sys

# Ajouter la racine du projet au sys.path
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

# Invalider le cache mémoire des modules rfdetr au cas où le script est exécuté dans un kernel Jupyter/Colab persistant
for _m in list(sys.modules.keys()):
    if _m.startswith("rfdetr"):
        del sys.modules[_m]

from rfdetr.data.sam_refiner import (
    refine_entire_coco_dataset,
    run_sam_preview_10_samples,
)

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s: %(message)s")
logger = logging.getLogger("sam_cli")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Raffinement des bounding boxes mammographiques via SAM (Segment Anything Model)"
    )
    parser.add_argument(
        "--json-file", "--input-json", dest="json_file", required=True,
        help="Chemin vers le fichier JSON COCO d'entrée"
    )
    parser.add_argument(
        "--images-dir", default=None,
        help="Dossier contenant les images mammographiques (détecté automatiquement si omis)"
    )
    parser.add_argument(
        "--num-samples", type=int, default=10,
        help="Nombre de clichés à tester dans le mode aperçu (défaut : 10)"
    )
    parser.add_argument(
        "--model-type", choices=["vit_b", "vit_l", "vit_h"], default="vit_b",
        help="Architecture SAM : 'vit_b' (375MB, recommandé T4), 'vit_l', 'vit_h' (défaut : vit_b)"
    )
    parser.add_argument(
        "--sam-checkpoint", default=None,
        help="Chemin vers le fichier de poids .pth (téléchargé automatiquement si omis)"
    )
    default_save_dir = "/content" if os.path.exists("/content") else "./sam_preview"
    parser.add_argument(
        "--save-dir", default=default_save_dir,
        help=f"Dossier où enregistrer la figure comparative (défaut : {default_save_dir})"
    )
    parser.add_argument(
        "--show", action="store_true", default=True,
        help="Afficher la figure directement dans la sortie du notebook Colab/Jupyter"
    )
    parser.add_argument(
        "--full-dataset", action="store_true", default=False,
        help="Traiter l'intégralité du dataset et exporter un nouveau JSON COCO"
    )
    parser.add_argument(
        "--output-json", default=None,
        help="Chemin de sortie pour le nouveau JSON COCO (requis si --full-dataset)"
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Graine aléatoire pour sélectionner les 10 images (défaut : 42)"
    )
    parser.add_argument(
        "--split-multi-lesions", action="store_true", default=True,
        help="Découper les nodules multiples d'une boîte en sous-boîtes distinctes (Option A)"
    )
    parser.add_argument(
        "--no-split", dest="split_multi_lesions", action="store_false",
        help="Conserver une boîte unique même si plusieurs nodules sont présents (Option B)"
    )
    return parser.parse_args()


def main():
    args = parse_args()

    if args.full_dataset:
        if not args.output_json:
            default_out = args.json_file.replace(".json", "_tight.json")
            if default_out == args.json_file:
                default_out = args.json_file + ".tight.json"
            args.output_json = default_out
            logger.info(f"--output-json non spécifié, utilisation automatique de : {args.output_json}")

        refine_entire_coco_dataset(
            input_json_path=args.json_file,
            output_json_path=args.output_json,
            images_dir=args.images_dir,
            model_type=args.model_type,
            checkpoint_path=args.sam_checkpoint,
            split_multi_lesions=args.split_multi_lesions,
        )
    else:
        results = run_sam_preview_10_samples(
            json_path=args.json_file,
            images_dir=args.images_dir,
            num_samples=args.num_samples,
            model_type=args.model_type,
            checkpoint_path=args.sam_checkpoint,
            save_dir=args.save_dir,
            show=args.show,
            seed=args.seed,
            split_multi_lesions=args.split_multi_lesions,
        )

        preview_img = results["preview_image"]
        print(f"\n💡 Pour inspecter la figure dans une cellule Google Colab :")
        print(f"    from IPython.display import Image, display")
        print(f"    display(Image('{preview_img}'))\n")


if __name__ == "__main__":
    main()
