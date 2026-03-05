# DDSP - Refactored for Optional CUDA Support

## Changes Made

### 1. CUDA Support is Now Optional ✓
CUDA libraries are no longer hard dependencies. Install them only if you need GPU support:

```bash
# CPU-only installation (default)
pip install ddsp
# or with uv
uv pip install ddsp

# With GPU/CUDA support
pip install ddsp[cuda]
# or with uv
uv pip install ddsp[cuda]

# Auto-detect and install with CUDA if GPU available
python check_cuda.py --install
```

### 2. Crepe Build Fixed ✓
The `crepe` package build issue has been resolved by:
- Adding `build-constraint-dependencies` in `[tool.uv]` section
- Setting `build-isolation = false` for legacy package compatibility
- Including `setuptools<70` in dev/test extras

**Seamless installation now works:**
```bash
uv sync --extra test
```

### 3. Python Version Updated
- **Before:** Python >=3.10,<3.12
- **After:** Python >=3.11,<3.13

Supported versions: 3.11, 3.12

### 4. Dependency Updates
- TensorFlow: >=2.15.0,<2.18 (was <2.16)
- tensorflow-probability: >=0.22.0,<0.25 (was <0.23)

### 5. UV Support Enhanced
Project now includes comprehensive UV configuration:

```bash
# Using uv (recommended)
uv venv --python 3.11
uv sync --extra dev --extra cuda  # with CUDA
uv sync --extra dev               # CPU only
```

## Installation Guide

### Quick Start (CPU)
```bash
pip install ddsp
```

### With GPU Support
```bash
# Option 1: Explicit CUDA install
pip install ddsp[cuda]

# Option 2: Auto-detect
git clone https://github.com/arsenylosev/ddsp.git
cd ddsp
python check_cuda.py --install
```

### Development Setup with UV (Recommended)
```bash
git clone https://github.com/arsenylosev/ddsp.git
cd ddsp
uv venv --python 3.11
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# For CPU-only development:
uv sync --extra dev

# For GPU development:
uv sync --extra dev --extra cuda
```

### Legacy pip Setup
```bash
git clone https://github.com/arsenylosev/ddsp.git
cd ddsp
pip install -e ".[dev]"      # CPU
pip install -e ".[dev,cuda]" # GPU
```

## CUDA Auto-Detection

The `check_cuda.py` script detects GPU availability:

```bash
# Check only
python check_cuda.py

# Check and auto-install
python check_cuda.py --install
```

This script checks for:
1. TensorFlow GPU detection
2. `nvidia-smi` availability
3. NVIDIA driver presence

## Verifying Installation

```python
import ddsp
print(ddsp.__version__)

# Check GPU availability (works on both CPU and GPU)
import tensorflow as tf
print(f"GPUs available: {len(tf.config.list_physical_devices('GPU'))}")
```

## Backward Compatibility

This change is **backward compatible** for functionality:
- The library works identically on CPU-only machines
- GPU functionality is available when `ddsp[cuda]` is installed
- Existing code requires no changes

**Breaking change:** Users who previously relied on automatic CUDA installation must now explicitly install with `pip install ddsp[cuda]`.

## Troubleshooting

### Crepe Build Issues
If you encounter `pkg_resources` errors during crepe build:
```bash
pip install "setuptools<70" wheel
uv sync --extra test
```

The pyproject.toml now includes build constraints that should handle this automatically.

### CUDA Not Detected
If CUDA is installed but not detected:
```bash
# Verify CUDA installation
nvidia-smi

# Force CUDA install
pip install ddsp[cuda]
```

## GitHub Actions CI

The CI workflow now uses UV for faster builds and includes the crepe build fixes.

```yaml
# Example CI step
- name: Install dependencies
  run: |
    uv venv --python 3.11
    uv sync --extra data_preparation --extra test
```
