#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
tuning_reference="${1:-outputs/behavior-v1}"
tuning_output="${2:-outputs/history-tuning-v1}"
tuning_archive="${3:-history-tuning-results.tar.gz}"
tuning_minutes="${4:-90}"
export PYTHONUNBUFFERED=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export OMP_NUM_THREADS=2
export OPENBLAS_NUM_THREADS=2
export MKL_NUM_THREADS=2
python -c 'import torch; assert torch.cuda.is_available(), "CUDA unavailable"; print(torch.cuda.get_device_name(0), flush=True)'
python -m spend_tuning plan
mkdir -p "$tuning_output"
python -u -m spend_tuning run --reference "$tuning_reference" --out-dir "$tuning_output" \
  --device cuda --threads 2 --max-minutes "$tuning_minutes" --resume 2>&1 | tee -a "$tuning_output/console.log"
if [[ ! -e "$tuning_archive" ]]; then
  python -m spend_tuning export --out-dir "$tuning_output" --archive "$tuning_archive"
else
  echo "Existing archive preserved: $tuning_archive; export to a new filename for an updated snapshot"
fi
