"""rfdetr.solver.checkpoint

Persistent Best Checkpointer hook for Detectron2 / Detrex.
Prevents overwriting 'model_best.pth' when resuming training across Colab session restarts.
Automatically tracks historical best validation metrics from 'metrics.json' or persistent JSON metadata.
"""

from __future__ import annotations

import json
import logging
import math
import os
import shutil
import time
from typing import Any, Dict, Optional, Tuple

try:
    from detectron2.engine.hooks import BestCheckpointer, HookBase
except ImportError:
    BestCheckpointer = object
    HookBase = object

logger = logging.getLogger("rfdetr.checkpoint")


def find_historical_best_metric(
    output_dir: str,
    val_metric: str = "bbox/AP50",
    mode: str = "max",
) -> Tuple[Optional[float], Optional[int]]:
    """Scanne le fichier metrics.json de Detectron2 pour retrouver le meilleur score historique.

    Permet de restaurer immédiatement le record précédent même si le fichier de métadonnées
    n'avait pas encore été créé avant le crash ou le redémarrage de la session Colab.

    Returns:
        Tuple: (best_score, best_iteration) ou (None, None) si non trouvé.
    """
    metrics_file = os.path.join(output_dir, "metrics.json")
    if not os.path.isfile(metrics_file):
        return None, None

    best_val: Optional[float] = None
    best_it: Optional[int] = None

    try:
        with open(metrics_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                    if val_metric in record:
                        val = float(record[val_metric])
                        if math.isnan(val):
                            continue
                        it = int(record.get("iteration", -1))
                        if best_val is None:
                            best_val = val
                            best_it = it
                        elif mode == "max" and val > best_val:
                            best_val = val
                            best_it = it
                        elif mode == "min" and val < best_val:
                            best_val = val
                            best_it = it
                except (json.JSONDecodeError, ValueError):
                    continue
    except Exception as e:
        logger.warning(f"Impossible de scanner metrics.json: {e}")

    return best_val, best_it


class PersistentBestCheckpointer(HookBase):
    """Hook de sauvegarde du meilleur modèle résistant aux redémarrages de session Colab.

    Contrairement au BestCheckpointer standard de Detectron2 qui réinitialise best_metric à -inf
    à chaque nouveau processus Python, ce hook :
      1. Charge le meilleur score enregistré dans '{file_prefix}_metric.json' s'il existe.
      2. Si absent, recherche automatiquement dans 'metrics.json' le meilleur score jamais atteint.
      3. Refuse formellement d'écraser 'model_best.pth' si le nouveau score est inférieur au record.
      4. Crée une copie de secours '{file_prefix}_backup.pth' avant tout écrasement.
    """

    def __init__(
        self,
        eval_period: int,
        checkpointer: Any,
        val_metric: str = "bbox/AP50",
        mode: str = "max",
        file_prefix: str = "model_best",
        output_dir: Optional[str] = None,
    ):
        self._eval_period = eval_period
        self._checkpointer = checkpointer
        self._val_metric = val_metric
        assert mode in ["max", "min"], f"mode doit être 'max' ou 'min', reçu: {mode}"
        self._mode = mode
        self._file_prefix = file_prefix

        # Répertoire de sortie
        self._output_dir = output_dir or getattr(checkpointer, "save_dir", "./output")
        self._meta_file = os.path.join(self._output_dir, f"{self._file_prefix}_metric.json")

        self._best_metric: Optional[float] = None
        self._best_iter: Optional[int] = None

        # Initialisation du record à partir du disque
        self._restore_best_metric_from_disk()

    def _restore_best_metric_from_disk(self) -> None:
        """Restaure le meilleur score précédent depuis le fichier JSON dédié ou depuis metrics.json."""
        # 1. Vérifier le fichier JSON persistant model_best_metric.json
        if os.path.isfile(self._meta_file):
            try:
                with open(self._meta_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                val = float(data.get("best_score", float("nan")))
                it = int(data.get("iteration", -1))
                if not math.isnan(val):
                    self._best_metric = val
                    self._best_iter = it
                    logger.info(
                        f"🛡️ [PersistentBestCheckpointer] Record restauré depuis {Path(self._meta_file).name} : "
                        f"{self._val_metric} = {self._best_metric:.5f} @ iteration {self._best_iter}"
                    )
                    return
            except Exception as e:
                logger.warning(f"Erreur lors de la lecture de {self._meta_file}: {e}")

        # 2. Si absent, scanner metrics.json dans output_dir
        hist_val, hist_it = find_historical_best_metric(
            output_dir=self._output_dir,
            val_metric=self._val_metric,
            mode=self._mode,
        )
        if hist_val is not None:
            self._best_metric = hist_val
            self._best_iter = hist_it
            logger.info(
                f"📈 [PersistentBestCheckpointer] Record historique retrouvé dans metrics.json : "
                f"{self._val_metric} = {self._best_metric:.5f} @ iteration {self._best_iter}"
            )
            # Sauvegarder ce record pour les prochains redémarrages
            self._persist_best_metric(self._best_metric, self._best_iter)
        else:
            logger.info(
                f"ℹ️ [PersistentBestCheckpointer] Aucun record précédent trouvé dans {self._output_dir}. "
                f"Initialisation à {'-∞' if self._mode == 'max' else '+∞'}."
            )

    def _persist_best_metric(self, val: float, iteration: int) -> None:
        """Écrit le nouveau record sur le disque."""
        try:
            os.makedirs(self._output_dir, exist_ok=True)
            with open(self._meta_file, "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "val_metric": self._val_metric,
                        "best_score": float(val),
                        "iteration": int(iteration),
                        "mode": self._mode,
                        "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                    },
                    f,
                    indent=2,
                )
        except Exception as e:
            logger.warning(f"Impossible d'écrire {self._meta_file}: {e}")

    def _update_best(self, val: float, iteration: int) -> bool:
        """Compare la valeur courante au record et retourne True si un nouveau record est battu."""
        if math.isnan(val):
            logger.warning(f"⚠️ Score d'évaluation {self._val_metric} est NaN à l'itération {iteration}. Ignoré.")
            return False

        # Si aucun record n'existait, initialiser
        if self._best_metric is None:
            self._best_metric = val
            self._best_iter = iteration
            self._persist_best_metric(val, iteration)
            return True

        is_better = (val > self._best_metric) if self._mode == "max" else (val < self._best_metric)

        if is_better:
            prev_best = self._best_metric
            prev_iter = self._best_iter
            self._best_metric = val
            self._best_iter = iteration

            # Sauvegarde de secours de l'ancien model_best.pth avant écrasement
            best_pth = os.path.join(self._output_dir, f"{self._file_prefix}.pth")
            if os.path.isfile(best_pth):
                backup_pth = os.path.join(self._output_dir, f"{self._file_prefix}_prev.pth")
                try:
                    shutil.copy2(best_pth, backup_pth)
                except Exception:
                    pass

            self._persist_best_metric(val, iteration)
            logger.info(
                f"🏆 NOUVEAU RECORD BATTU pour {self._val_metric} : {val:.5f} (précédent: {prev_best:.5f} @ iter {prev_iter}) @ iteration {iteration} !"
            )
            return True
        else:
            logger.info(
                f"🔒 [PersistentBestCheckpointer] Score actuel {self._val_metric} = {val:.5f} @ iter {iteration} "
                f"n'est pas meilleur que le record de {self._best_metric:.5f} @ iter {self._best_iter}. "
                f"'{self._file_prefix}.pth' conservé intact."
            )
            return False

    def after_step(self) -> None:
        next_iter = self.trainer.iter + 1
        is_final = next_iter == self.trainer.max_iter
        if is_final or (self._eval_period > 0 and next_iter % self._eval_period == 0):
            # Vérifier si la métrique est présente dans le stockage
            if not hasattr(self.trainer, "storage") or not self.trainer.storage.iter_has_history(self._val_metric, next_iter):
                return
            latest_val = float(self.trainer.storage.history(self._val_metric).latest())
            if self._update_best(latest_val, next_iter):
                self._checkpointer.save(f"{self._file_prefix}")
                logger.info(f"💾 Checkpoint '{self._file_prefix}.pth' mis à jour avec succès !")
