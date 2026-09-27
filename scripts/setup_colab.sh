#!/usr/bin/env bash
# Universal Environment Setup for RF-DETR on Google Colab / Remote GPU Instances
set -eo pipefail

echo "=========================================================="
echo "  RF-DETR: Automated Environment Setup & Detrex Compilation"
echo "=========================================================="

# 1. Hardware Verification
python3 -c "
import torch
assert torch.cuda.is_available(), '❌ Error: No NVIDIA GPU detected.'
name = torch.cuda.get_device_name(0)
major, minor = torch.cuda.get_device_capability(0)
print(f'✅ GPU: {name} (Compute Capability: {major}.{minor})')
assert major >= 7, '❌ Error: GPU compute capability must be >= 7.0 (V100, T4, A100, etc.)'
"

# 2. Derive CUDA Architecture & Compilation Concurrency
TARGET_ARCH=$(python3 -c "import torch; major, minor = torch.cuda.get_device_capability(0); print(f'{major}.{minor}')")
TOTAL_RAM_GB=$(python3 -c "import psutil; print(f'{psutil.virtual_memory().total / (1024**3):.1f}')")
MAX_JOBS=$(python3 -c "import psutil, multiprocessing; print('1' if (psutil.virtual_memory().total / (1024**3)) < 16.0 else str(min(2, multiprocessing.cpu_count())))")

echo "🎯 Targeted CUDA Arch: sm_${TARGET_ARCH//./} (${TARGET_ARCH})"
echo "⚙️ System RAM: ${TOTAL_RAM_GB} GB | Compilation MAX_JOBS: ${MAX_JOBS}"

# 3. System Packages
echo "📦 Installing system build utilities (ninja-build)..."
apt-get update -qq && apt-get install -y -qq ninja-build

# 4. Pinned Python Dependencies
echo "📦 Installing Python dependencies..."
pip install -q timm==1.0.29 scipy opencv-python pycocotools matplotlib tensorboard tensorboardX fairscale einops omegaconf fvcore iopath psutil

# 5. Detectron2 Installation
echo "📦 Installing Detectron2 from git..."
pip install -q 'git+https://github.com/facebookresearch/detectron2.git'

# 6. Detrex Compilation & Installation
if [ ! -d "detrex" ]; then
    echo "Cloning Detrex repository..."
    git clone https://github.com/IDEA-Research/detrex.git
fi


echo "Compiling Detrex CUDA extensions (MSDeformAttn)..."
cd detrex
rm -rf build/ detrex/_C*.so detrex.egg-info
MAX_JOBS="${MAX_JOBS}" CUDA_HOME=/usr/local/cuda FORCE_CUDA=1 TORCH_CUDA_ARCH_LIST="${TARGET_ARCH}" python3 setup.py build_ext --inplace

# Expose Detrex via .pth file for Python 3.12 / 3.13 compatibility (avoiding deprecated setup.py develop)
PYTHON_SITE=$(python3 -c "import site; print(site.getsitepackages()[0])")
echo "$(pwd)" > "${PYTHON_SITE}/detrex.pth"
pip install --no-build-isolation --no-deps -e . 2>/dev/null || true
cd ..

# 7. Symlink Detrex for absolute path compatibility (/content/detrex)
if [ -d "/content" ] && [ ! -e "/content/detrex" ]; then
    ln -s "$(pwd)/detrex" /content/detrex 2>/dev/null || true
fi

# 8. Install RF-DETR in Editable Mode
echo "📦 Installing RF-DETR package..."
echo "$(pwd)" > "${PYTHON_SITE}/rfdetr.pth"
pip install --no-build-isolation --no-deps -e . 2>/dev/null || pip install -e . 2>/dev/null || true

echo "=========================================================="
echo "✅ RF-DETR Installation Completed Successfully!"
echo "=========================================================="

