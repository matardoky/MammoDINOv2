"""rfdetr.models.dino

This module intentionally defers to the official detrex DINO implementation.

The DINO detection head (transformer encoder/decoder, Hungarian Matcher,
DINOCriterion, Contrastive DeNoising, positional encodings) is provided
directly by detrex — no reimplementation needed.

Usage in config (LazyConfig):
    from detrex.modeling.models import DINO
    from detrex.modeling.matcher import HungarianMatcher
    from detrex.modeling.criterion import SetCriterion  # or DINO's built-in

See: https://github.com/IDEA-Research/detrex/tree/main/projects/dino
"""

try:
    from projects.dino.modeling import DINO
except ImportError:
    raise ImportError(
        "detrex is required for the DINO detection head. "
        "Install it via: pip install git+https://github.com/IDEA-Research/detrex.git "
        "or follow the Colab setup in scripts/setup_colab.sh"
    )

__all__ = ["DINO"]
