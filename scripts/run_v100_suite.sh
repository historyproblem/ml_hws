#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
suite_config="${1:-configs/v100_suite.json}"
suite_output="${2:-outputs/v100-suite}"
export PYTHONUNBUFFERED=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export OMP_NUM_THREADS=2
export OPENBLAS_NUM_THREADS=2
export MKL_NUM_THREADS=2
python -c 'import torch; assert torch.cuda.is_available(), "CUDA unavailable"; print("GPU:", torch.cuda.get_device_name(0), "Torch:", torch.__version__, flush=True)'
python -m spend_baselines.suite plan --config "$suite_config"
mkdir -p "$suite_output"
python -u -m spend_baselines.suite run --config "$suite_config" --out-dir "$suite_output" --device cuda --threads 2 --resume 2>&1 | tee -a "$suite_output/console.log"
