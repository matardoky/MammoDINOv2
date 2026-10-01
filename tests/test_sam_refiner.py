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
