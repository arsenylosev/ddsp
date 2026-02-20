#!/usr/bin/env bash

# Install DDSP package in editable mode when conda environment is activated
# This registers console scripts: ddsp_run, ddsp_prepare_tfrecord, ddsp_export, etc.

set -euo pipefail

# Absolute path to the DDSP repository (adjust if repository moves)
DDSP_REPO_DIR="/mnt/d2/data_sda/nfs/alosev/AI_Synthesizer/experimental_work/ddsp"

echo "[ddsp-activate] Installing DDSP and training dependencies..."

# Install required dependencies for training console scripts
pip install --no-deps -e "${DDSP_REPO_DIR}"
pip install google-cloud-storage cloudml-hypertune note_seq tqdm apache_beam pytz 'setuptools<70.0.0'

echo "[ddsp-activate] DDSP and dependencies installed successfully."

# Verify console scripts are available
if command -v ddsp_run &>/dev/null; then
  echo "[ddsp-activate] ddsp_run command: OK"
else
  echo "[ddsp-activate] Warning: ddsp_run command not found."
fi

if command -v ddsp_prepare_tfrecord &>/dev/null; then
  echo "[ddsp-activate] ddsp_prepare_tfrecord command: OK"
else
  echo "[ddsp-activate] Warning: ddsp_prepare_tfrecord command not found."
fi
