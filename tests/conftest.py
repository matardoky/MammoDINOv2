"""tests/conftest.py

Global pytest configuration and warning filters for RF-DETR test suite.
Suppresses upstream deprecation warnings from detrex, timm, and pkg_resources
to ensure clean test reports.
"""

import warnings

# Suppress known upstream warnings from detrex/timm/torch
warnings.filterwarnings("ignore", category=FutureWarning, message=r".*torch\.cuda\.amp\.custom_.*")
warnings.filterwarnings("ignore", category=FutureWarning, module=r".*timm\.models\.layers.*")
warnings.filterwarnings("ignore", category=FutureWarning, module=r".*detrex.*")
warnings.filterwarnings("ignore", category=FutureWarning, module=r".*detectron2\.layers\.dcn_v3.*")
warnings.filterwarnings("ignore", category=UserWarning, message=r".*pkg_resources is deprecated.*")
