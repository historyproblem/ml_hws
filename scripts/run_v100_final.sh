#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONUNBUFFERED=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export OMP_NUM_THREADS=2
export OPENBLAS_NUM_THREADS=2
export MKL_NUM_THREADS=2
mkdir -p outputs/final-checks-20261115
python -u scripts/run_final_checks.py finish \
  --config configs/final_checks.json \
  --device cuda --threads 2 --resume \
  --out-dir outputs/final-checks-20261115 "$@" \
  2>&1 | tee -a outputs/final-checks-20261115/console.log
