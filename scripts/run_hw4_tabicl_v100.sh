#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
HW4_ICL_OUT="${1:-analysis/hw4-tabicl-v100-01}"
HW4_ICL_RAW="${2:-data}"
HW4_ICL_HOST="${HW4_PYTHON:-python}"
HW4_ICL_ASSETS="${HW4_TABICL_ASSETS:-analysis/.hw4-tabicl-v2-assets}"
HW4_ICL_VENV="$HW4_ICL_ASSETS/venv"
mkdir -p "$HW4_ICL_ASSETS"
exec 9>"$HW4_ICL_ASSETS/setup-run.lock"
flock -n 9 || { echo "A TabICL setup/run already holds this environment." >&2; exit 1; }
"$HW4_ICL_HOST" - <<'PY'
import torch
if not torch.cuda.is_available():
    raise SystemExit("Activate the original HW4 Python with a working CUDA torch")
print({"torch":torch.__version__,"gpu":torch.cuda.get_device_name(0)})
PY
"$HW4_ICL_HOST" -m venv --system-site-packages "$HW4_ICL_VENV"
"$HW4_ICL_HOST" - "$HW4_ICL_VENV" <<'PY'
import sys,sysconfig,subprocess
from pathlib import Path
v=Path(sys.argv[1]).resolve()
local=Path(subprocess.check_output([str(v/"bin/python"),"-c","import sysconfig;print(sysconfig.get_path('purelib'))"],text=True).strip())
if v not in local.parents or v==Path(sys.prefix):
    raise SystemExit("Require a separate TabICL environment")
(local/"original_hw4_host.pth").write_text("\n".join(dict.fromkeys(sysconfig.get_path(k) for k in ("purelib","platlib")))+"\n")
PY
HW4_ICL_PY="$HW4_ICL_VENV/bin/python"
HW4_ICL_WHEEL="$HW4_ICL_ASSETS/tabicl-2.2.0-py3-none-any.whl"
if [[ ! -f "$HW4_ICL_WHEEL" ]]; then
 "$HW4_ICL_PY" -m pip download --disable-pip-version-check --no-deps --only-binary=:all: --dest "$HW4_ICL_ASSETS" 'tabicl==2.2.0'
fi
"$HW4_ICL_PY" - "$HW4_ICL_WHEEL" <<'PY'
import sys,hashlib
from pathlib import Path
if hashlib.sha256(Path(sys.argv[1]).read_bytes()).hexdigest()!="c80b8ec2719be26bc0bf1dd758a2361f58019d8f7877038b785529a425789ce4":
    raise SystemExit("TabICL wheel checksum changed")
PY
HW4_ICL_STAMP="$HW4_ICL_ASSETS/environment-ready.sha256"
HW4_ICL_HASH=$("$HW4_ICL_HOST" - <<'PY'
import hashlib,sys,torch
from pathlib import Path
h=hashlib.sha256(Path("scripts/run_hw4_tabicl_v100.sh").read_bytes())
h.update(Path("configs/hw4-tabicl-v2-overlay.txt").read_bytes())
h.update((sys.executable+" "+torch.__version__).encode());print(h.hexdigest())
PY
)
if [[ ! -f "$HW4_ICL_STAMP" ]] || [[ "$(cat "$HW4_ICL_STAMP")" != "$HW4_ICL_HASH" ]]; then
 "$HW4_ICL_PY" -m pip install --disable-pip-version-check --no-deps --only-binary=:all: --ignore-installed "$HW4_ICL_WHEEL" -r configs/hw4-tabicl-v2-overlay.txt
 echo "$HW4_ICL_HASH" > "$HW4_ICL_STAMP"
fi
HW4_ICL_MODEL="$HW4_ICL_ASSETS/tabicl-classifier-v2-20260212.ckpt"
"$HW4_ICL_PY" - "$HW4_ICL_MODEL" <<'PY'
import shutil,sys,hashlib
from pathlib import Path
from urllib.request import urlopen
p=Path(sys.argv[1]);expected="bdc7dbd5e4ff21f8f0456fcf90c6b7cdf72dbea960f2d05b19bec19f9b3d4ed0"
if not p.exists():
    tmp=p.with_suffix(".partial")
    with urlopen("https://huggingface.co/jingang/TabICL/resolve/main/tabicl-classifier-v2-20260212.ckpt",timeout=120) as source,tmp.open("wb") as target:
        shutil.copyfileobj(source,target,1<<20)
    if hashlib.sha256(tmp.read_bytes()).hexdigest()!=expected: raise SystemExit("Unexpected downloaded checksum")
    tmp.replace(p)
if hashlib.sha256(p.read_bytes()).hexdigest()!=expected: raise SystemExit("Unexpected cached checksum")
PY
export PYTHONHASHSEED=42 CUBLAS_WORKSPACE_CONFIG=:4096:8 HF_HUB_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 VECLIB_MAXIMUM_THREADS=2
exec "$HW4_ICL_PY" -u scripts/run_hw4_tabicl_v100.py --raw "$HW4_ICL_RAW" --out "$HW4_ICL_OUT" \
 --checkpoint "$HW4_ICL_MODEL" --device cuda --threads 2 --resume
