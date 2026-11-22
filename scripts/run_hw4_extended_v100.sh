#!/usr/bin/env bash
# Three independent GPU experiments; preserve every completed return archive.
set -euo pipefail
HW4_EXTENDED_ENTRY="$(cd "$(dirname "$0")" && pwd)/$(basename "$0")"
cd "$(dirname "$HW4_EXTENDED_ENTRY")/.."

HW4_EXTENDED_OUT="${1:-analysis/hw4-extended-v100-01}"
HW4_EXTENDED_RAW="${2:-data}"
HW4_EXTENDED_PYTHON="${HW4_PYTHON:-python}"
mkdir -p "$HW4_EXTENDED_OUT"

if [[ "${HW4_EXTENDED_LOCK_HELD:-0}" != "1" ]]; then
  export HW4_EXTENDED_LOCK_HELD=1
  exec "$HW4_EXTENDED_PYTHON" - "$HW4_EXTENDED_ENTRY" "$HW4_EXTENDED_OUT" "$HW4_EXTENDED_RAW" <<'PY'
from pathlib import Path
import fcntl
import subprocess
import sys
with (Path(sys.argv[2]) / '.launcher.lock').open('a') as lock:
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        raise SystemExit('This extended GPU run is already active; do not launch a duplicate.')
    result = subprocess.run(['bash', sys.argv[1], sys.argv[2], sys.argv[3]], pass_fds=(lock.fileno(),))
    raise SystemExit(result.returncode)
PY
fi

HW4_TABM_STATUS="complete"
if ! bash scripts/run_hw4_tabm_preprocessing_v100.sh "$HW4_EXTENDED_OUT/tabm" "$HW4_EXTENDED_RAW"; then
  HW4_TABM_STATUS="failed"
fi
HW4_PFN_STATUS="complete"
if ! bash scripts/run_hw4_tabpfn_v100.sh "$HW4_EXTENDED_OUT/tabpfn" "$HW4_EXTENDED_RAW"; then
  HW4_PFN_STATUS="failed"
fi
HW4_ICL_STATUS="complete"
if ! bash scripts/run_hw4_tabicl_v100.sh "$HW4_EXTENDED_OUT/tabicl" "$HW4_EXTENDED_RAW"; then
  HW4_ICL_STATUS="failed"
fi

"$HW4_EXTENDED_PYTHON" - "$HW4_EXTENDED_OUT" "$HW4_PFN_STATUS" "$HW4_ICL_STATUS" "$HW4_TABM_STATUS" <<'PY'
from pathlib import Path
import hashlib
import json
import sys
import tarfile

out = Path(sys.argv[1])
files = {}
for family, name in [('tabm', 'tabm_results.tar.gz'), ('tabpfn', 'tabpfn_results.tar.gz'), ('tabicl', 'tabicl_results.tar.gz')]:
    source = out / family / 'return_results.tar.gz'
    if source.is_file():
        files[name] = dict(source=source, sha256=hashlib.sha256(source.read_bytes()).hexdigest())
manifest = dict(format='hw4-extended-v100-bundle-1', tabpfn_status=sys.argv[2],
                tabicl_status=sys.argv[3], tabm_status=sys.argv[4],
                files={name: item['sha256'] for name, item in files.items()},
                kaggle_uploads=0, recent_quarter_metrics_computed=0,
                recent_predictions='Unscored; local one-shot frozen-finalist diagnostic only')
path = out / 'bundle-manifest.json'
path.write_text(json.dumps(manifest, indent=2) + '\n')
with tarfile.open(out / 'return_results.tar.gz', 'w:gz') as archive:
    archive.add(path, arcname='bundle-manifest.json')
    for name, item in files.items():
        archive.add(item['source'], arcname=name)
print(json.dumps(dict(event='extended_complete', archive=str(out / 'return_results.tar.gz'), **manifest)), flush=True)
PY

if [[ "$HW4_TABM_STATUS" != "complete" || "$HW4_PFN_STATUS" != "complete" || "$HW4_ICL_STATUS" != "complete" ]]; then
  exit 1
fi
