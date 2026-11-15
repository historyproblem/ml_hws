#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
embedding_run="${1:-outputs/item-embeddings-v1}"
history_run="${2:-outputs/history-tuning-v1}"
embedding_reference="${3:-outputs/behavior-v1}"
export PYTHONUNBUFFERED=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export OMP_NUM_THREADS=2
export OPENBLAS_NUM_THREADS=2
export MKL_NUM_THREADS=2
# Validate/reuse the existing history refits; never train them again here.
python scripts/export_joint_history_submission.py --run-dir "$history_run" \
  --reference "$embedding_reference" --output "$embedding_run/baseline_validation.csv"
python -c 'import torch; assert torch.cuda.is_available(), "CUDA unavailable"; print(torch.cuda.get_device_name(0), flush=True)'
python -u -m spend_embeddings refit --profile item32 --reference "$embedding_reference" \
  --out-dir "$embedding_run" --device cuda --threads 2 --max-minutes 0 --resume
python scripts/export_item32_submission.py --run-dir "$embedding_run" \
  --baseline-dir "$history_run" --reference "$embedding_reference"
