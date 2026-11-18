#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
HW4_PYTHON="${PYTHON_BIN:-python}"
export OMP_NUM_THREADS="${HW4_THREADS:-4}"
export OPENBLAS_NUM_THREADS="$OMP_NUM_THREADS"
export MKL_NUM_THREADS="$OMP_NUM_THREADS"
exec "$HW4_PYTHON" -u scripts/run_hw4_classical.py \
  --raw "${2:-data}" --out "${1:-analysis/hw4-classical-01}" \
  --threads "$OMP_NUM_THREADS" --resume
