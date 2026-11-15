#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
submission_run="${1:-outputs/history-tuning-v1}"
submission_reference="${2:-outputs/behavior-v1}"
export PYTHONUNBUFFERED=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export OMP_NUM_THREADS=2
export OPENBLAS_NUM_THREADS=2
export MKL_NUM_THREADS=2
python -c 'import torch; assert torch.cuda.is_available(), "CUDA unavailable"; print(torch.cuda.get_device_name(0), flush=True)'
python -u -m spend_tuning refit --profile baseline --reference "$submission_reference" \
  --out-dir "$submission_run" --device cuda --threads 2 --max-minutes 0 --resume
python scripts/export_joint_history_submission.py --run-dir "$submission_run" --reference "$submission_reference"
