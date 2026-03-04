#!/bin/bash
# GPU activation script for DDSP uv environment
# Source this file to enable GPU support: source .venv/bin/activate_gpu.sh

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Activate the uv environment
source "$SCRIPT_DIR/activate"

# Set protobuf workaround
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python

# Add conda CUDA libraries to LD_LIBRARY_PATH for GPU support
if [ -d "/home/jovyan/.uconda/envs/ddsp_env/lib" ]; then
    export LD_LIBRARY_PATH=/home/jovyan/.uconda/envs/ddsp_env/lib:/usr/local/nvidia/lib:/usr/local/nvidia/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
fi

echo "DDSP environment activated with GPU support"
echo "CUDA libraries: $LD_LIBRARY_PATH"
