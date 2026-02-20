#!/usr/bin/env bash

# Optional cleanup when conda environment is deactivated.
# This script currently uninstalls the DDSP package to keep the environment clean.
# Comment out the uninstall block if you prefer DDSP to persist across activations.

set -euo pipefail

if pip show ddsp >/dev/null 2>&1; then
  echo "[ddsp-deactivate] Uninstalling DDSP package..."
  pip uninstall -y ddsp >/dev/null 2>&1 || true
  echo "[ddsp-deactivate] DDSP package removed."
else
  echo "[ddsp-deactivate] DDSP package not installed; nothing to uninstall."
fi
