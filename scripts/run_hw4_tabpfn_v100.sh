#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

HW4_PFN_OUT="${1:-analysis/hw4-tabpfn-v100-01}"
HW4_PFN_RAW="${2:-data}"
HW4_PFN_HOST="${HW4_PYTHON:-python}"
HW4_PFN_ASSETS="${HW4_TABPFN_ASSETS:-analysis/.hw4-tabpfn-v2-assets}"
HW4_PFN_VENV="$HW4_PFN_ASSETS/venv"
mkdir -p "$HW4_PFN_ASSETS"
exec 9>"$HW4_PFN_ASSETS/setup-run.lock"
flock -n 9 || { echo "Another TabPFN setup or run is active for this environment." >&2; exit 1; }

"$HW4_PFN_HOST" - <<'PY'
import sys,torch
if not torch.cuda.is_available():
    raise SystemExit("CUDA unavailable in original HW4 Python. Activate the V100 environment.")
print({"host":sys.executable,"torch":torch.__version__,"gpu":torch.cuda.get_device_name(0)})
PY
"$HW4_PFN_HOST" -m venv --system-site-packages "$HW4_PFN_VENV"

# A nested venv does not normally inherit packages installed in the parent
# venv. Append only its site-package directories so the native CUDA torch and
# existing scientific libraries remain visible, without modifying the parent.
"$HW4_PFN_HOST" - "$HW4_PFN_VENV" <<'PY'
from pathlib import Path
import subprocess,sys,sysconfig
v=Path(sys.argv[1]).resolve()
local=Path(subprocess.check_output([str(v/"bin/python"),"-c","import sysconfig; print(sysconfig.get_path('purelib'))"],text=True).strip())
if v not in local.parents or v==Path(sys.prefix):
    raise SystemExit("Overlay must be a distinct virtual environment")
paths=dict.fromkeys(str(Path(sysconfig.get_path(k)).resolve()) for k in ("purelib","platlib"))
(local/"hw4_cuda_host_packages.pth").write_text("\n".join(paths)+"\n")
PY
HW4_PFN_PY="$HW4_PFN_VENV/bin/python"
HW4_PFN_WHEEL="$HW4_PFN_ASSETS/tabpfn-2.2.1-py3-none-any.whl"
if [[ ! -f "$HW4_PFN_WHEEL" ]]; then
  "$HW4_PFN_PY" -m pip download --disable-pip-version-check --no-deps --only-binary=:all: --dest "$HW4_PFN_ASSETS" 'tabpfn==2.2.1'
fi
"$HW4_PFN_PY" - "$HW4_PFN_WHEEL" <<'PY'
import hashlib,sys
from pathlib import Path
if hashlib.sha256(Path(sys.argv[1]).read_bytes()).hexdigest()!="9aa3a0df8ecebcae5f426b1fec9bb2e2a690880bc8ad079191e8d959bf0f157b":
    raise SystemExit("TabPFN 2.2.1 wheel digest mismatch")
PY
# --ignore-installed writes each pinned overlay distribution locally, even if
# the shared host exposes another version. --no-deps prevents torch resolution.
HW4_PFN_READY="$HW4_PFN_ASSETS/environment-ready.sha256"
HW4_PFN_ENV_HASH=$("$HW4_PFN_HOST" - <<'PY'
import hashlib,sys,torch
from pathlib import Path
h=hashlib.sha256(Path("configs/hw4-tabpfn-v2-overlay.txt").read_bytes())
h.update(Path("scripts/run_hw4_tabpfn_v100.sh").read_bytes())
h.update((sys.executable+" "+torch.__version__).encode())
print(h.hexdigest())
PY
)
if [[ ! -f "$HW4_PFN_READY" ]] || [[ "$(cat "$HW4_PFN_READY")" != "$HW4_PFN_ENV_HASH" ]]; then
  "$HW4_PFN_PY" -m pip install --disable-pip-version-check --only-binary=:all: --no-deps --ignore-installed \
    "$HW4_PFN_WHEEL" -r configs/hw4-tabpfn-v2-overlay.txt
  echo "$HW4_PFN_ENV_HASH" > "$HW4_PFN_READY"
fi

HW4_PFN_CKPT="$HW4_PFN_ASSETS/tabpfn-v2-classifier-v2_default.ckpt"
"$HW4_PFN_PY" - "$HW4_PFN_CKPT" <<'PY'
from pathlib import Path
from urllib.request import urlopen
import hashlib,shutil,sys
path=Path(sys.argv[1])
expected="cf8c519c01eaf1613ee91239006d57b1c806ff5f23ac1aeb1315ba1015210e49"
if not path.exists():
    temp=path.with_suffix(".partial")
    with urlopen("https://huggingface.co/Prior-Labs/TabPFN-v2-clf/resolve/main/tabpfn-v2-classifier-v2_default.ckpt",timeout=90) as source, temp.open("wb") as dest:
        shutil.copyfileobj(source,dest,1024*1024)
    if hashlib.sha256(temp.read_bytes()).hexdigest()!=expected: raise SystemExit("Downloaded checkpoint digest mismatch")
    temp.replace(path)
if hashlib.sha256(path.read_bytes()).hexdigest()!=expected: raise SystemExit("Cached checkpoint digest mismatch")
license_path=path.with_name("LICENSE.PriorLabs-TabPFN.txt")
license_digest="08b1851b4b6136b5fb0cb124d3258e0d91e3e1623f1be5904b77e4d270d112d8"
if not license_path.exists():
    temp=license_path.with_suffix(".partial")
    with urlopen("https://huggingface.co/Prior-Labs/TabPFN-v2-clf/resolve/main/LICENSE.txt",timeout=45) as source:
        original=source.read()
    if hashlib.sha256(original).hexdigest()!=license_digest: raise SystemExit("License text changed; review it before using this release")
    temp.write_bytes(original)
    temp.replace(license_path)
if hashlib.sha256(license_path.read_bytes()).hexdigest()!=license_digest: raise SystemExit("Cached license digest mismatch")
print("Verified public V2 checkpoint and license. Built with PriorLabs-TabPFN. Inference will run offline.")
PY
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export TABPFN_DISABLE_TELEMETRY=1 HF_HUB_DISABLE_TELEMETRY=1 HF_HUB_OFFLINE=1
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 VECLIB_MAXIMUM_THREADS=2
exec "$HW4_PFN_PY" -u scripts/run_hw4_tabpfn_v100.py --raw "$HW4_PFN_RAW" --out "$HW4_PFN_OUT" \
  --checkpoint "$HW4_PFN_CKPT" --device cuda --threads 2 --resume
