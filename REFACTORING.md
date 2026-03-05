# DDSP - Refactored for Optional CUDA Support

## Changes Made

### 1. CUDA Support is Now Optional
CUDA libraries are no longer hard dependencies. Install them only if you need GPU support:

```bash
# CPU-only installation (default)
pip install ddsp

# With GPU/CUDA support
pip install ddsp[cuda]
# or
pip install ddsp[gpu]

# Development installation with all extras
pip install ddsp[dev,cuda]
```

### 2. Python Version Updated
- **Before:** Python >=3.10,<3.12
- **After:** Python >=3.11,<3.13

Supported versions: 3.11, 3.12

### 3. Dependency Updates
- TensorFlow: >=2.15.0,<2.18 (was <2.16)
- tensorflow-probability: >=0.22.0,<0.25 (was <0.23)

### 4. UV Support
Project now includes UV configuration for faster dependency resolution:

```bash
# Using uv
uv pip install -e ".[dev,cuda]"

# Or sync with lock file
uv sync
```

## Installation Guide

### Quick Start (CPU)
```bash
pip install ddsp
```

### With GPU Support
```bash
# Ensure you have CUDA-capable GPU
pip install ddsp[cuda]
```

### Development Setup
```bash
git clone https://github.com/arsenylosev/ddsp.git
cd ddsp
pip install -e ".[dev]"
```

### Using UV (Recommended)
```bash
git clone https://github.com/arsenylosev/ddsp.git
cd ddsp
uv venv --python 3.11
source .venv/bin/activate
uv pip install -e ".[dev,cuda]"
```

## Verifying Installation

```python
import ddsp
print(ddsp.__version__)

# Check GPU availability (will work on both CPU and GPU)
import tensorflow as tf
print(f"GPUs available: {len(tf.config.list_physical_devices('GPU'))}")
```

## Backward Compatibility

This change is **backward compatible** for functionality:
- The library works identically on CPU-only machines
- GPU functionality is available when `ddsp[cuda]` is installed
- Existing code requires no changes

**Breaking change:** Users who previously relied on automatic CUDA installation must now explicitly install with `pip install ddsp[cuda]`.
