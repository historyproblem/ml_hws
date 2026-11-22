#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

HW4_RUN_OUT="${1:-analysis/hw4-tabm-preprocessing-v100-01}"
HW4_RAW_DIR="${2:-data}"
HW4_PYTHON="${HW4_PYTHON:-python}"

"$HW4_PYTHON" - <<'PY'
import importlib.metadata as m
import pathlib
import torch
print({'torch':torch.__version__, 'cuda_available':torch.cuda.is_available()})
if not torch.cuda.is_available():
    raise SystemExit('CUDA unavailable. Select the V100 environment; no CPU fallback.')
print({'gpu':torch.cuda.get_device_name(0)})
PY

# Add only the two small pinned official libraries. Torch/CUDA/scientific
# packages are preserved: this command never resolves/upgrades dependencies.
"$HW4_PYTHON" -m pip install --no-deps -r configs/hw4-tabm-requirements.txt
exec "$HW4_PYTHON" -u scripts/run_hw4_tabm_preprocessing_v100.py \
  --raw "$HW4_RAW_DIR" --out "$HW4_RUN_OUT" --device cuda --threads 4 --resume
