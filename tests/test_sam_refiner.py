"""tests/test_sam_refiner.py

Tests unitaires pour le module de raffinement d'annotations SAM (rfdetr/data/sam_refiner.py).
Fonctionne entièrement sur CPU et données synthétiques sans nécessiter segment-anything ou CUDA.
"""

import subprocess
import sys
from pathlib import Path
import numpy as np
import pytest

from rfdetr.data.sam_refiner import refine_box_from_mask

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def test_refine_box_from_mask_tightens_correctly():
    """Vérifie qu'un masque binaire centré resserre correctement la boîte originale."""
    # Masque 100x100 avec un objet carré de 40x40 au centre [30..70, 30..70]
    mask = np.zeros((100, 100), dtype=bool)
    mask[30:70, 30:70] = True

    # Boîte originale lâche de 80x80 [10, 10, 80, 80]
    original_box = [10.0, 10.0, 80.0, 80.0]  # aire = 6400

    tight_box, reduction_pct = refine_box_from_mask(mask, original_box)

    # La nouvelle boîte doit être [30, 30, 40, 40] (aire = 1600)
    assert tight_box[0] == pytest.approx(30.0, abs=1.0)
    assert tight_box[1] == pytest.approx(30.0, abs=1.0)
    assert tight_box[2] == pytest.approx(40.0, abs=1.0)
    assert tight_box[3] == pytest.approx(40.0, abs=1.0)

    # Réduction d'aire attendue : (1 - 1600/6400) * 100 = 75%
    assert reduction_pct == pytest.approx(75.0, abs=1.0)


def test_refine_box_from_mask_empty_mask_fallback():
    """Vérifie que si le masque est vide, la boîte originale est conservée (fallback)."""
    mask = np.zeros((100, 100), dtype=bool)
    original_box = [20.0, 25.0, 50.0, 60.0]

    tight_box, reduction_pct = refine_box_from_mask(mask, original_box)
    assert tight_box == original_box
    assert reduction_pct == 0.0


def test_refine_box_from_mask_tiny_noise_fallback():
    """Vérifie qu'un masque trop petit (<10% de l'aire) déclenche le fallback de sécurité."""
    mask = np.zeros((100, 100), dtype=bool)
    mask[40:42, 40:42] = True  # aire = 4 px²
    original_box = [10.0, 10.0, 80.0, 80.0]  # aire = 6400 px² (ratio < 0.1%)

    tight_box, reduction_pct = refine_box_from_mask(mask, original_box, min_area_ratio=0.10)
    # Déclenche le fallback
    assert tight_box == original_box
    assert reduction_pct == 0.0


def test_sam_cli_help():
    """Vérifie que le script CLI scripts/refine_annotations_with_sam.py s'exécute avec --help."""
    script_path = PROJECT_ROOT / "scripts" / "refine_annotations_with_sam.py"
    res = subprocess.run(
        [sys.executable, str(script_path), "--help"],
        capture_output=True,
        text=True,
        cwd=str(PROJECT_ROOT),
    )
    assert res.returncode == 0, f"refine_annotations_with_sam.py --help failed: {res.stderr}"
    stdout = res.stdout
    assert "--json-file" in stdout
    assert "--images-dir" in stdout
    assert "--num-samples" in stdout
    assert "--model-type" in stdout
    assert "--full-dataset" in stdout


def test_resolve_image_path_nested_and_case_insensitive(tmp_path):
    """Vérifie la robustesse de resolve_image_path face aux sous-dossiers et à la casse."""
    from rfdetr.utils.visualize import resolve_image_path, _IMAGE_DIR_INDEX

    _IMAGE_DIR_INDEX.clear()

    # Création d'une structure de dossiers
    sub_dir = tmp_path / "subfolder" / "nested"
    sub_dir.mkdir(parents=True)
    img_file = sub_dir / "18792357_3468574009796872_L_CC_6f25c293d6.PNG"
    img_file.write_bytes(b"dummy")

    # 1. Résolution avec chemin relatif et casse différente (.png vs .PNG)
    found = resolve_image_path("18792357_3468574009796872_L_CC_6f25c293d6.png", images_fallback_dir=str(tmp_path))
    assert found is not None
    assert Path(found).resolve() == img_file.resolve()

    # 2. Résolution avec sous-dossier non mentionné
    found2 = resolve_image_path("nested/18792357_3468574009796872_L_CC_6f25c293d6.PNG", images_fallback_dir=str(tmp_path))
    assert found2 is not None

    # 3. Fichier inexistant
    assert resolve_image_path("unknown_file_123.png", images_fallback_dir=str(tmp_path)) is None


def test_standalone_sam_preview_cli_help():
    """Vérifie que le script standalone scripts/standalone_sam_preview.py s'exécute avec --help."""
    script_path = PROJECT_ROOT / "scripts" / "standalone_sam_preview.py"
    res = subprocess.run(
        [sys.executable, str(script_path), "--help"],
        capture_output=True,
        text=True,
        cwd=str(PROJECT_ROOT),
    )
    assert res.returncode == 0, f"standalone_sam_preview.py --help failed: {res.stderr}"
    stdout = res.stdout
    assert "--json-file" in stdout
    assert "--images-dir" in stdout
    assert "--num-samples" in stdout
    assert "--split-multi-lesions" in stdout


def test_extract_refined_boxes_multi_component_split():
    """Vérifie que 2 nodules distincts dans une même boîte sont bien scindés en 2 boîtes (Option A)."""
    from rfdetr.data.sam_refiner import extract_refined_boxes_from_mask

    # Masque 200x200 avec 2 nodules distincts
    mask = np.zeros((200, 200), dtype=bool)
    # Nodule 1 : 20x20 pixels [20..40, 20..40] (aire 400)
    mask[20:40, 20:40] = True
    # Nodule 2 : 25x25 pixels [140..165, 140..165] (aire 625)
    mask[140:165, 140:165] = True

    # Grande boîte englobante originale de 180x180 (aire = 32400)
    orig_box = [10.0, 10.0, 180.0, 180.0]

    # Avec splitting (Option A)
    results = extract_refined_boxes_from_mask(mask, orig_box, split_multi=True)
    assert len(results) == 2, f"Attendu 2 boîtes, obtenu {len(results)}"
    assert results[0]["is_split"] is True
    assert results[1]["is_split"] is True

    # Le premier nodule retourné est le plus grand (Nodule 2)
    b0 = results[0]["box"]
    assert b0[0] == pytest.approx(140.0, abs=1.0)
    assert b0[1] == pytest.approx(140.0, abs=1.0)
    assert b0[2] == pytest.approx(25.0, abs=1.0)
    assert b0[3] == pytest.approx(25.0, abs=1.0)

    # Le deuxième nodule retourné est Nodule 1
    b1 = results[1]["box"]
    assert b1[0] == pytest.approx(20.0, abs=1.0)
    assert b1[1] == pytest.approx(20.0, abs=1.0)
    assert b1[2] == pytest.approx(20.0, abs=1.0)
    assert b1[3] == pytest.approx(20.0, abs=1.0)


def test_extract_refined_boxes_no_split():
    """Vérifie que sans splitting (Option B), une seule boîte englobant les 2 nodules est produite."""
    from rfdetr.data.sam_refiner import extract_refined_boxes_from_mask

    mask = np.zeros((200, 200), dtype=bool)
    mask[20:40, 20:40] = True
    mask[140:165, 140:165] = True
    orig_box = [10.0, 10.0, 180.0, 180.0]

    # Sans splitting (Option B)
    results = extract_refined_boxes_from_mask(mask, orig_box, split_multi=False)
    assert len(results) == 1
    assert results[0]["is_split"] is False
    # La boîte commune s'étend de x=20 à x=165 (largeur 145)
    b = results[0]["box"]
    assert b[0] == pytest.approx(20.0, abs=1.0)
    assert b[2] == pytest.approx(145.0, abs=1.0)


def test_generate_sam_datasets_cli_help():
    """Vérifie que scripts/generate_sam_datasets.py s'exécute avec --help."""
    script_path = PROJECT_ROOT / "scripts" / "generate_sam_datasets.py"
    res = subprocess.run(
        [sys.executable, str(script_path), "--help"],
        capture_output=True,
        text=True,
        cwd=str(PROJECT_ROOT),
    )
    assert res.returncode == 0, f"generate_sam_datasets.py --help failed: {res.stderr}"
    stdout = res.stdout
    assert "--train-json" in stdout
    assert "--val-json" in stdout
    assert "--images-dir" in stdout




