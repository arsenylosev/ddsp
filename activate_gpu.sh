#!/bin/bash
# GPU activation script for DDSP uv environment
# Source this file to enable GPU support: source activate_gpu.sh

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

source "$SCRIPT_DIR/.venv/bin/activate"

export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python

find_pip_cuda_libs() {
    local venv_path="$SCRIPT_DIR/.venv"
    local python_version=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
    local site_packages="$venv_path/lib/python${python_version}/site-packages"

    local cuda_paths=""
    for pkg in cublas cudnn cuda_nvrtc cuda_runtime cufft curand cusolver cusparse nvjitlink nvtx; do
        if [ -d "$site_packages/nvidia/$pkg/lib" ]; then
            cuda_paths="$site_packages/nvidia/$pkg/lib${cuda_paths:+:$cuda_paths}"
        fi
    done

    echo "$cuda_paths"
}

find_pip_cuda_nvvm() {
    local venv_path="$SCRIPT_DIR/.venv"
    local python_version=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
    local site_packages="$venv_path/lib/python${python_version}/site-packages"

    if [ -d "$site_packages/nvidia/cuda_nvcc/nvvm/libdevice" ]; then
        echo "$site_packages/nvidia/cuda_nvcc/nvvm"
    fi
}

PIP_CUDA_PATHS=$(find_pip_cuda_libs)
PIP_CUDA_NVVM=$(find_pip_cuda_nvvm)

if [ -n "$PIP_CUDA_PATHS" ]; then
    export LD_LIBRARY_PATH="$PIP_CUDA_PATHS${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
    echo "GPU: Using pip-installed CUDA libraries"

    if [ -n "$PIP_CUDA_NVVM" ]; then
        export CUDA_ROOT="$PIP_CUDA_NVVM"
    fi

    if [ -d "/usr/local/cuda/lib64" ]; then
        export LD_LIBRARY_PATH="/usr/local/cuda/lib64:$LD_LIBRARY_PATH"
    fi
elif [ -d "/usr/local/cuda/lib64" ]; then
    export LD_LIBRARY_PATH="/usr/local/cuda/lib64:/usr/local/nvidia/lib:/usr/local/nvidia/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
    export CUDA_ROOT="/usr/local/cuda"
    echo "GPU: Using system CUDA libraries"
else
    echo "GPU: Warning - No CUDA libraries detected. GPU may not be available."
fi

echo "DDSP environment activated with GPU support"
