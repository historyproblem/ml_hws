#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

# Activate the existing CUDA environment first, or set PYTHON_BIN explicitly.
HW4_PYTHON="${PYTHON_BIN:-python}"
export CUBLAS_WORKSPACE_CONFIG=:4096:8
"$HW4_PYTHON" - <<'PY'
import json
import torch
if not torch.cuda.is_available():
    raise RuntimeError("A CUDA environment is required")
props = torch.cuda.get_device_properties(0)
print(json.dumps({'torch': torch.__version__, 'gpu': props.name,
                  'memory_gib': props.total_memory / 1024**3}))
PY

case "${1:-history}" in
  history)
    "$HW4_PYTHON" scripts/run_hw4_neural.py --device cuda \
      --models mlp visit_transformer --batch-size 256 --raw "${3:-data}" \
      --out "${2:-analysis/hw4-neural-v100-history}"
    ;;
  features)
    "$HW4_PYTHON" scripts/run_hw4_neural.py --device cuda \
      --models feature_transformer --batch-size 32 --raw "${3:-data}" \
      --out "${2:-analysis/hw4-neural-v100-features}"
    ;;
  *)
    echo "Usage: bash scripts/run_hw4_neural_v100.sh {history|features} [new-output-directory] [csv-directory]" >&2
    exit 2
    ;;
esac
