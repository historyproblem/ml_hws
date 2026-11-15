#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONUNBUFFERED=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export OMP_NUM_THREADS=2
export OPENBLAS_NUM_THREADS=2
export MKL_NUM_THREADS=2
python -c 'import torch; assert torch.cuda.is_available(), "CUDA unavailable"; print(torch.cuda.get_device_name(0), flush=True)'
# A valid fallback exists even if the search hits its time budget.
bash scripts/make_course10_submission.sh
python scripts/tune_course10_fast.py plan
mkdir -p outputs/course10-fast-tuning
python -u scripts/tune_course10_fast.py run --workers "${1:-2}" --max-minutes "${2:-65}" \
  --device cuda --threads 2 --resume 2>&1 | tee -a outputs/course10-fast-tuning/console.log
