#!/usr/bin/env bash
# Create the Linux CUDA environment and build the MS-DeformAttn extension.
set -euo pipefail

PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_FILE="${ENV_FILE:-${PROJECT_ROOT}/environment.yml}"
ENV_NAME="${ENV_NAME:-tree-detr}"
CUDA_ARCH="${TORCH_CUDA_ARCH_LIST:-8.6}"

if ! command -v conda >/dev/null 2>&1; then
    echo "ERROR: conda was not found in PATH." >&2
    exit 1
fi
if [[ ! -f "${ENV_FILE}" ]]; then
    echo "ERROR: environment file not found: ${ENV_FILE}" >&2
    exit 1
fi

eval "$(conda shell.bash hook)"
if conda env list | awk '{print $1}' | grep -Fxq "${ENV_NAME}"; then
    echo "Updating conda environment: ${ENV_NAME}"
    # Older copies of this environment may contain torchaudio 2.1.x. It pins
    # PyTorch 2.1.x and makes the requested PyTorch 2.4.1 solve impossible;
    # Tree-DETR does not use torchaudio.
    if conda list --name "${ENV_NAME}" | awk 'NR > 3 && $1 == "torchaudio" { found=1 } END { exit !found }'; then
        echo "Removing stale incompatible package: torchaudio"
        conda remove --name "${ENV_NAME}" --yes torchaudio
    fi
    conda env update --name "${ENV_NAME}" --file "${ENV_FILE}" --prune
else
    echo "Creating conda environment: ${ENV_NAME}"
    conda env create --name "${ENV_NAME}" --file "${ENV_FILE}"
fi

export TORCH_CUDA_ARCH_LIST="${CUDA_ARCH}"
conda activate "${ENV_NAME}"
# Compile with this environment's toolkit, not a host CUDA 13 installation.
export CUDA_HOME="${CONDA_PREFIX}"
export PATH="${CUDA_HOME}/bin:${PATH}"
cd "${PROJECT_ROOT}/models/ops"
python - <<'PY'
import re
import subprocess
import torch
import torchvision
from torch.utils.cpp_extension import CUDA_HOME

print('torch', torch.__version__, 'torchvision', torchvision.__version__)
print('torch CUDA', torch.version.cuda, 'CUDA_HOME', CUDA_HOME)
if not torch.cuda.is_available():
    raise SystemExit('ERROR: CUDA GPU is unavailable; check driver and GPU visibility.')
nvcc = subprocess.check_output([CUDA_HOME + '/bin/nvcc', '--version'], text=True)
print(nvcc)
release = re.search(r'release (\d+\.\d+)', nvcc)
if release is None or release.group(1) != torch.version.cuda:
    raise SystemExit('ERROR: nvcc and PyTorch CUDA versions must match.')
for index in range(torch.cuda.device_count()):
    print('GPU', index, torch.cuda.get_device_name(index), torch.cuda.get_device_capability(index))
PY
echo "Building MS-DeformAttn for CUDA arch ${TORCH_CUDA_ARCH_LIST}"
conda run --no-capture-output --name "${ENV_NAME}" \
    python "${PROJECT_ROOT}/models/ops/setup.py" build_ext --inplace

conda run --no-capture-output --name "${ENV_NAME}" \
    python -c "import MultiScaleDeformableAttention; print('MS-DeformAttn extension import OK')"

echo "Environment ready: ${ENV_NAME}"
echo "Activate with: conda activate ${ENV_NAME}"
