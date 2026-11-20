#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export CUBLAS_WORKSPACE_CONFIG=:4096:8
HW4_PYTHON="${PYTHON_BIN:-python}"
exec "$HW4_PYTHON" -u scripts/run_hw4_tabm.py --raw "${2:-data}" \
 --out "${1:-analysis/hw4-tabm-v100-01}" --device cuda --resume
