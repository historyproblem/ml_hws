#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
HW4_ICL32_OUT="${1:-analysis/hw4-tabicl-native32-01}"
HW4_ICL32_RAW="${2:-data}"
HW4_ICL32_ASSETS="${HW4_TABICL_ASSETS:-analysis/.hw4-tabicl-v2-assets}"
HW4_ICL32_PY="$HW4_ICL32_ASSETS/venv/bin/python"
HW4_ICL32_MODEL="$HW4_ICL32_ASSETS/tabicl-classifier-v2-20260212.ckpt"
[[ -x "$HW4_ICL32_PY" && -f "$HW4_ICL32_MODEL" ]] || {
 echo 'Existing completed TabICL V100 environment and public checkpoint required.' >&2; exit 1;
}
exec 9>"$HW4_ICL32_ASSETS/setup-run.lock"
flock -n 9 || { echo 'Another TabICL process holds this environment.' >&2; exit 1; }
export PYTHONHASHSEED=42 CUBLAS_WORKSPACE_CONFIG=:4096:8 HF_HUB_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 VECLIB_MAXIMUM_THREADS=2
exec "$HW4_ICL32_PY" -u scripts/run_hw4_tabicl_ensemble32.py --raw "$HW4_ICL32_RAW" \
 --out "$HW4_ICL32_OUT" --checkpoint "$HW4_ICL32_MODEL" --device cuda --threads 2 --resume
