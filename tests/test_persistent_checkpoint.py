"""tests/test_persistent_checkpoint.py

Unit tests for PersistentBestCheckpointer and find_historical_best_metric.
Verifies that resuming training never overwrites an existing best checkpoint with a lower score.
"""

import json
import pytest
from pathlib import Path
from rfdetr.solver.checkpoint import PersistentBestCheckpointer, find_historical_best_metric


def test_find_historical_best_metric(tmp_path):
    """Vérifie la détection du record absolu dans un fichier metrics.json."""
    metrics_file = tmp_path / "metrics.json"
    lines = [
        {"bbox/AP50": 15.2, "iteration": 1000},
        {"bbox/AP50": 23.96648, "iteration": 18000},
        {"bbox/AP50": 26.62839, "iteration": 19549},
        {"bbox/AP50": 20.1, "iteration": 21000},  # Score plus faible après
    ]
    with open(metrics_file, "w", encoding="utf-8") as f:
        for record in lines:
            f.write(json.dumps(record) + "\n")

    best_val, best_iter = find_historical_best_metric(str(tmp_path), val_metric="bbox/AP50")
    assert best_val == pytest.approx(26.62839, rel=1e-5)
    assert best_iter == 19549


def test_persistent_best_checkpointer_prevents_overwrite_on_resume(tmp_path):
    """Vérifie que PersistentBestCheckpointer refuse d'écraser model_best avec un score plus faible."""
    # Simuler le fichier metrics.json existant avant le crash de Colab
    metrics_file = tmp_path / "metrics.json"
    with open(metrics_file, "w", encoding="utf-8") as f:
        f.write(json.dumps({"bbox/AP50": 26.62839, "iteration": 19549}) + "\n")

    # Mock checkpointer
    saved_files = []
    class DummyCheckpointer:
        def __init__(self, save_dir):
            self.save_dir = save_dir
        def save(self, name):
            saved_files.append(name)

    dummy_cp = DummyCheckpointer(str(tmp_path))

    # Initialiser le hook (simule le redémarrage d'un nouveau processus Python)
    hook = PersistentBestCheckpointer(
        eval_period=391,
        checkpointer=dummy_cp,
        val_metric="bbox/AP50",
        mode="max",
        file_prefix="model_best",
        output_dir=str(tmp_path),
    )

    # Le record doit être immédiatement restauré à 26.62839
    assert hook._best_metric == pytest.approx(26.62839, rel=1e-5)
    assert hook._best_iter == 19549

    # 1. Évaluation à 21.0% (plus faible que le record)
    is_better = hook._update_best(21.0, 21500)
    assert is_better is False, "Un score inférieur ne doit PAS battre le record !"
    assert hook._best_metric == pytest.approx(26.62839, rel=1e-5)
    assert hook._best_iter == 19549

    # 2. Évaluation à 28.5% (vrai nouveau record)
    is_better_new = hook._update_best(28.5, 22000)
    assert is_better_new is True, "Un vrai meilleur score doit être validé !"
    assert hook._best_metric == pytest.approx(28.5, rel=1e-5)
    assert hook._best_iter == 22000

    # Vérifier que le fichier JSON persistant a bien été créé
    meta_file = tmp_path / "model_best_metric.json"
    assert meta_file.exists()
    with open(meta_file, "r", encoding="utf-8") as f:
        meta_data = json.load(f)
    assert meta_data["best_score"] == pytest.approx(28.5, rel=1e-5)
    assert meta_data["iteration"] == 22000
