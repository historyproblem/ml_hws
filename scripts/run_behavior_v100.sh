#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
behavior_reference="${1:-outputs/v100-focused}"
behavior_output="${2:-outputs/behavior-v1}"
behavior_phase="${3:-joint}"
case "$behavior_phase" in
  joint|scout|run) ;;
  *) echo "Phase must be joint, scout, or run" >&2; exit 2 ;;
esac
export PYTHONUNBUFFERED=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export OMP_NUM_THREADS=2
export OPENBLAS_NUM_THREADS=2
export MKL_NUM_THREADS=2
python -c 'import torch; assert torch.cuda.is_available(), "CUDA unavailable"; print(torch.cuda.get_device_name(0), flush=True)'
python -m spend_behavior plan
mkdir -p "$behavior_output"
behavior_options=()
if [[ -d "$behavior_reference/cache" ]]; then
  behavior_options+=(--base-cache "$behavior_reference/cache")
fi
python -u -m spend_behavior "$behavior_phase" --reference "$behavior_reference" \
  --out-dir "$behavior_output" --device cuda --threads 2 --resume \
  "${behavior_options[@]}" 2>&1 | tee -a "$behavior_output/console.log"
