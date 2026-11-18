#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export CUBLAS_WORKSPACE_CONFIG=:4096:8
HW4_PYTHON="${PYTHON_BIN:-python}"
exec "$HW4_PYTHON" -u scripts/run_hw4_residual.py \
  --device cuda --resume --out "${1:-analysis/hw4-residual-v100-01}" \
  --raw "${2:-data}" --classical-results "${3:-analysis/hw4-classical-01}"
