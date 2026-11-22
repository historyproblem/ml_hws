#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
HW4_BAG_OUT="${1:-analysis/hw4-tabicl-seedbag-01}"
HW4_BAG_RAW="${2:-data}"
HW4_BAG_BASE="${3:-analysis/hw4-tabicl-native32-01}"
HW4_BAG_ASSETS="${HW4_TABICL_ASSETS:-analysis/.hw4-tabicl-v2-assets}"
HW4_BAG_PY="$HW4_BAG_ASSETS/venv/bin/python"
HW4_BAG_MODEL="$HW4_BAG_ASSETS/tabicl-classifier-v2-20260212.ckpt"
[[ -x "$HW4_BAG_PY" && -f "$HW4_BAG_MODEL" && -f "$HW4_BAG_BASE/completion.json" ]] || {
 echo 'Completed existing TabICL environment/checkpoint/native32 seed42 directory required.' >&2; exit 1;
}
exec 9>"$HW4_BAG_ASSETS/setup-run.lock"
flock -n 9 || { echo 'Another TabICL run holds this environment.' >&2; exit 1; }
export PYTHONHASHSEED=42 CUBLAS_WORKSPACE_CONFIG=:4096:8 HF_HUB_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 VECLIB_MAXIMUM_THREADS=2
exec "$HW4_BAG_PY" -u scripts/run_hw4_tabicl_seedbag.py --raw "$HW4_BAG_RAW" --out "$HW4_BAG_OUT" \
 --baseline32-dir "$HW4_BAG_BASE" --checkpoint "$HW4_BAG_MODEL" --device cuda --threads 2 --resume
