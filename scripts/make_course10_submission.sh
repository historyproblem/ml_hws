#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
course_model="${1:-mlp}"
course_reference="${2:-outputs/v100-focused}"
course_output="${3:-outputs/course10-only}"
course_device="cpu"
if [[ "$course_model" == "mlp" ]]; then
  course_device="cuda"
  python -c 'import torch; assert torch.cuda.is_available(), "CUDA unavailable"; print(torch.cuda.get_device_name(0), flush=True)'
fi
export PYTHONUNBUFFERED=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export OMP_NUM_THREADS=2
export OPENBLAS_NUM_THREADS=2
export MKL_NUM_THREADS=2
python -m spend_course10 plan
python -m spend_course10 compare --reference "$course_reference" --out-dir "$course_output"
python -u -m spend_course10 refit --model "$course_model" --reference "$course_reference" \
  --out-dir "$course_output" --device "$course_device" --threads 2 --resume
