#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
suite_workers="${1:-2}"
suite_config="${2:-configs/v100_suite.json}"
suite_output="${3:-outputs/v100-focused}"
export PYTHONUNBUFFERED=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export OMP_NUM_THREADS=2
export OPENBLAS_NUM_THREADS=2
export MKL_NUM_THREADS=2
mkdir -p "$suite_output"
python -u scripts/run_suite_parallel.py --workers "$suite_workers" --config "$suite_config" --out-dir "$suite_output" --device cuda --threads 2 2>&1 | tee -a "$suite_output/parallel-console.log"
