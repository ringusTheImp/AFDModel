#!/bin/bash
# ============================================================
# WX-AFD: Derecho Environment Setup (idempotent — safe to re-run)
# ============================================================
set -euo pipefail

# ---- Config ----
WX_AFD_ROOT="/glade/derecho/scratch/$USER/wx-afd"
CONDA_ENV="wx-afd"
PROJECT_CODE="<PROJECT_CODE>"   # Replace with your NCAR project code
SOURCE_DIR="$(cd "$(dirname "$0")" && pwd)"  # Directory containing this script

echo "=========================================="
echo "  WX-AFD Derecho Setup"
echo "  Root: $WX_AFD_ROOT"
echo "  Env:  $CONDA_ENV"
echo "=========================================="

# ---- 1. Module loads ----
echo ""
echo ">>> Loading modules..."
module --force purge
module load ncarenv/24.12
module load cuda/12.8.0
module load conda
echo "  Modules loaded."

# ---- 2. Conda environment ----
echo ""
echo ">>> Setting up conda environment '$CONDA_ENV'..."

# Ensure conda shell functions are available (needed in non-interactive shells)
eval "$(conda shell.bash hook)"

if conda env list | grep -q "^${CONDA_ENV} "; then
    echo "  Environment '$CONDA_ENV' already exists, skipping creation."
else
    echo "  Creating environment '$CONDA_ENV' with Python 3.11..."
    conda create -n "$CONDA_ENV" python=3.11 -y
    echo "  Environment created."
fi
conda activate "$CONDA_ENV"
echo "  Activated: $(python --version)"

# ---- 3. Pip installs ----
echo ""
echo ">>> Installing packages..."
pip install --quiet torch --index-url https://download.pytorch.org/whl/cu121
pip install --quiet axolotl==0.14.0
pip install --quiet flash-attn --no-build-isolation
pip install --quiet rouge-score bert-score sacrebleu tqdm
echo "  Packages installed."

# ---- 4. Directory structure ----
echo ""
echo ">>> Creating directory structure..."
mkdir -p "$WX_AFD_ROOT"/{data,configs,output,eval,scripts,logs}
echo "  Directories created at $WX_AFD_ROOT"

# ---- 5. Copy data and configs (skip if already present) ----
echo ""
echo ">>> Copying project files..."

# Data files
for f in train.jsonl val.jsonl; do
    src="$SOURCE_DIR/data/$f"
    dst="$WX_AFD_ROOT/data/$f"
    if [ -f "$src" ]; then
        cp -n "$src" "$dst" 2>/dev/null && echo "  Copied $f" || echo "  $f already exists, skipped."
    else
        echo "  WARNING: $src not found (run 03_build_dataset.py first)"
    fi
done

# Config
src="$SOURCE_DIR/configs/wx-afd-dora.yml"
dst="$WX_AFD_ROOT/configs/wx-afd-dora.yml"
if [ -f "$src" ]; then
    cp -n "$src" "$dst" 2>/dev/null && echo "  Copied wx-afd-dora.yml" || echo "  Config already exists, skipped."
fi

# Scripts
for f in evaluate.py generate.py train.pbs eval.pbs; do
    src="$SOURCE_DIR/scripts/$f"
    dst="$WX_AFD_ROOT/scripts/$f"
    if [ -f "$src" ]; then
        cp -n "$src" "$dst" 2>/dev/null && echo "  Copied $f" || echo "  $f already exists, skipped."
    fi
done

# Shared module
src="$SOURCE_DIR/wx_afd.py"
dst="$WX_AFD_ROOT/wx_afd.py"
if [ -f "$src" ]; then
    cp -n "$src" "$dst" 2>/dev/null && echo "  Copied wx_afd.py" || echo "  wx_afd.py already exists, skipped."
fi

# ---- 6. (Optional) Pre-download model ----
echo ""
echo ">>> Pre-downloading model (compute nodes may lack internet)..."
if command -v huggingface-cli &>/dev/null; then
    huggingface-cli download Qwen/Qwen3-4B-Instruct-2507 --quiet || \
        echo "  Model download failed (may need HF_TOKEN or network access)"
else
    echo "  huggingface-cli not found, skipping model pre-download."
    echo "  Install with: pip install huggingface_hub[cli]"
fi

# ---- 7. Verification ----
echo ""
echo "=========================================="
echo "  Verification"
echo "=========================================="

python -c "
import torch
print(f'PyTorch:       {torch.__version__}')
print(f'CUDA avail:    {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU:           {torch.cuda.get_device_name(0)}')

import axolotl
print(f'Axolotl:       {axolotl.__version__}')

from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained('Qwen/Qwen3-4B-Instruct-2507', trust_remote_code=True)
eos_id = tok.convert_tokens_to_ids('<|im_end|>')
pad_id = tok.convert_tokens_to_ids('<|endoftext|>')
print(f'EOS token ID:  {eos_id} (expect 151645)')
print(f'PAD token ID:  {pad_id} (expect 151643)')
assert eos_id == 151645, f'EOS mismatch: {eos_id}'
assert pad_id == 151643, f'PAD mismatch: {pad_id}'
assert eos_id != pad_id, 'EOS and PAD must differ!'
print('Token IDs:     OK')
"

# Check data files
echo ""
for f in train.jsonl val.jsonl; do
    p="$WX_AFD_ROOT/data/$f"
    if [ -f "$p" ]; then
        count=$(wc -l < "$p")
        echo "Data:          $f — $count examples"
    else
        echo "Data:          $f — NOT FOUND"
    fi
done

# Check config
if [ -f "$WX_AFD_ROOT/configs/wx-afd-dora.yml" ]; then
    echo "Config:        wx-afd-dora.yml — OK"
else
    echo "Config:        wx-afd-dora.yml — NOT FOUND"
fi

echo ""
echo "=========================================="
echo "  Setup complete!"
echo "  Next: qsub scripts/train.pbs"
echo "=========================================="
