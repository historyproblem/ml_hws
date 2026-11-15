#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
embedding_reference="${1:-outputs/behavior-v1}"
embedding_output="${2:-outputs/item-embeddings-v1}"
embedding_archive="${3:-item-embeddings-results.tar.gz}"
embedding_minutes="${4:-90}"
export PYTHONUNBUFFERED=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export OMP_NUM_THREADS=2
export OPENBLAS_NUM_THREADS=2
export MKL_NUM_THREADS=2
python -c 'import torch; assert torch.cuda.is_available(), "CUDA unavailable"; print(torch.cuda.get_device_name(0), flush=True)'
python -m spend_embeddings plan
mkdir -p "$embedding_output"
python -u -m spend_embeddings run --reference "$embedding_reference" --out-dir "$embedding_output" \
  --device cuda --threads 2 --max-minutes "$embedding_minutes" --resume 2>&1 | tee -a "$embedding_output/console.log"
if [[ ! -e "$embedding_archive" ]]; then
  python -m spend_embeddings export --out-dir "$embedding_output" --archive "$embedding_archive"
else
  echo "Existing archive preserved: $embedding_archive; export to a new filename for an updated snapshot"
fi
