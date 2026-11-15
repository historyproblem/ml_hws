#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
suite_date="${1:-2018-10-01}"
suite_output="${2:-outputs/v100-focused}"
export PYTHONUNBUFFERED=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export OMP_NUM_THREADS=2
export OPENBLAS_NUM_THREADS=2
export MKL_NUM_THREADS=2
mkdir -p "$suite_output"
python -u scripts/run_suite_parallel.py --coverage-only --dates "$suite_date" --bootstrap-members 4 --workers 1 --out-dir "$suite_output" --device cuda --threads 2 2>&1 | tee -a "$suite_output/finish-console.log"
