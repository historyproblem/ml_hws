#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
HW4_PYTHON="${PYTHON_BIN:-python}"
exec "$HW4_PYTHON" -u scripts/export_hw4_final.py --raw "${2:-data}" \
  --out "${1:-outputs/hw4/final-20261120}" --device "${3:-cpu}" --resume
