# AGENTS.md - DDSP Development Guide

AI agent guidelines for the DDSP (Differentiable Digital Signal Processing) codebase.

## Environment Setup

**Python**: 3.11 (upgraded from 3.10, TensorFlow 2.15 requires Python 3.10-3.11)

**Installation with uv** (Recommended - 10-100x faster):
```bash
# Install uv if not already installed
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create and activate virtual environment
uv venv --python 3.11
source .venv/bin/activate  # Linux/Mac
# .venv\Scripts\activate   # Windows

# Install with all development dependencies
uv sync --extra data_preparation --extra test

# Or install in editable mode for development
uv pip install -e ".[data_preparation,test]"
```

**GPU Support**: To enable GPU support, you need CUDA libraries. Use the provided activation script:
```bash
# This automatically sets up CUDA libraries from conda and protobuf workaround
source ./activate_gpu.sh

# Or manually set these environment variables:
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
export LD_LIBRARY_PATH=/home/jovyan/.uconda/envs/ddsp_env/lib:/usr/local/nvidia/lib:/usr/local/nvidia/lib64:$LD_LIBRARY_PATH
```

**Alternative: Installation with pip**:
```bash
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
pip install -e ".[data_preparation,test]"
```

**Legacy: Conda Installation** (for CUDA/GPU support):
```bash
conda create -n ddsp_env python=3.11 -y
conda activate ddsp_env
conda install -c conda-forge cudnn=8.9 cuda-toolkit=12.5 -y
pip install -e ".[data_preparation,test]"
```

**Console Scripts**: `ddsp_export`, `ddsp_run`, `ddsp_prepare_tfrecord`, `ddsp_generate_synthetic_dataset`, `ddsp_ai_platform`

**Required Workaround for Protobuf**: For TFDS + protobuf 6.x incompatibility with note_seq:
```bash
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
```

**Important**: Always run ddsp commands with the protobuf workaround:
```bash
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python && ddsp_run --mode=train ...
```

## Build/Lint/Test Commands

### Testing
```bash
# Using uv (recommended)
uv run pytest                                  # all tests
uv run pytest ddsp/core_test.py               # single file
uv run pytest ddsp/core_test.py::test_midi_to_hz_is_accurate  # single test
uv run pytest ddsp/spectral_ops_test.py
uv run pytest -k "test_name" ddsp/            # filter tests by name pattern
uv run pytest -x ddsp/                         # stop on first failure
uv run pytest --tb=short ddsp/                 # shorter traceback

# Using pip/conda
pytest                                  # all tests
pytest ddsp/core_test.py               # single file
pytest ddsp/core_test.py::test_midi_to_hz_is_accurate  # single test
PYTHONPATH=$(pwd):$PYTHONPATH pytest ddsp/spectral_ops_test.py
```

### Linting
```bash
# Using uv (recommended)
uv run ruff check ddsp                            # entire codebase
uv run ruff check --select E,W,F ddsp              # errors only
uv run ruff check ddsp/core.py                     # single file

# Using pip/conda
ruff check ddsp
ruff check --select E,W,F ddsp
ruff check ddsp/core.py
```

### Formatting
```bash
# Using uv (recommended)
uv run ruff format ddsp                            # format all files
uv run ruff format --check ddsp                     # check formatting (CI)
uv run ruff format ddsp/core.py                     # format single file

# Using pip/conda
ruff format ddsp
ruff format --check ddsp
ruff format ddsp/core.py
```

### Type Checking
```bash
# Using uv
uv run mypy ddsp
uv run mypy ddsp/core.py --ignore-missing-imports

# Using pip
mypy ddsp
mypy ddsp/core.py --ignore-missing-imports
```

### Running ddsp_run
```bash
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python && uv run ddsp_run \
  --mode=train \
  --save_dir=/path/to/experiments/test \
  --gin_file=ddsp/training/gin/models/solo_instrument.gin \
  --gin_file=ddsp/training/gin/datasets/tfrecord.gin \
  --gin_param="TFRecordProvider.file_pattern='/absolute/path/to/*.tfrecord*'"
```

## Code Style Guidelines

### Copyright Header
```python
# Copyright 2024 The DDSP Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
```

### Imports
Order: standard library → third-party → local. Group by type with blank lines.
```python
from collections import abc
import copy
from typing import Any, Dict, Optional, Sequence, Text, TypeVar

import gin
import numpy as np
from scipy import fftpack
import tensorflow.compat.v2 as tf

from ddsp import core
```
- Use `import tensorflow.compat.v2 as tf`
- Alias: `tfkl = tf.keras.layers`, `tfd = tfp.distributions`
- Avoid `from X import *`

### Formatting (from pylintrc)
- **Line length**: 80 characters max
- **Indentation**: 2 spaces (not 4)
- **Indent after paren**: 4 spaces

### Naming Conventions
- Variables/functions: `snake_case`
- Classes: `PascalCase`
- Constants: `UPPER_SNAKE_CASE`
- Private members: `_private_method`

### Type Hints
```python
TensorDict = Dict[Text, tf.Tensor]
Number = TypeVar('Number', int, float, np.ndarray, tf.Tensor)
```

### Function Design
- Google-style docstrings (purpose, args, returns)
- Single responsibility principle
- Return early for error cases

### Error Handling
```python
raise ValueError(f'Keys: {keys} must be the same length as {x}')
raise NotImplementedError  # for abstract methods
```

### TensorFlow Conventions
- Use `tf.float32` explicitly
- Check `tf.executing_eagerly()` for eager/graph decisions
- Use `tf.function` for performance-critical code
- Prefer Keras layers (`tfkl`)
- Set `autocast=False` in custom layers

### Configuration with Gin
- Use `@gin.register` for functions/classes in DAG configs
- Use `@gin.configurable` for globally configurable functions
- Keep core library agnostic to gin where possible
- Gin files located in `ddsp/training/gin/`

### Test Files
- Naming: `*_test.py`
- Use `from absl.testing import parameterized`
- Base classes: `parameterized.TestCase, tf.test.TestCase`
- Tests in same directory as implementation

## Project Structure

```
ddsp/
  core.py              # Core DSP functions (utilities, shifts, filters)
  spectral_ops.py      # STFT, mel, mag spectrograms
  processors.py        # Synthesizer processors (additive, spectral, noise)
  training/
    ddsp_run.py        # Main training entry point
    gin/               # Gin configuration files
    models/            # Model definitions
    decoders/          # Decoder architectures
```

## Audio Conventions
- Sample rate: 16000 Hz (default)
- Frame sizes: 2048, 4096, 8192 (power of 2)
- Overlap: 0.75 (75%) standard for STFT
- MIDI range: 0-127, f0_hz: 0-20000 Hz
- Audio tensors: `[batch, samples]` or `[samples]` (squeeze channel dim)

## Known Issues
1. **absl.flags conflict**: `prepare_tfrecord_lib_test.py` fails with pytest `-v`. Run without `-v`.
2. **NumPy version**: Use `numpy<2` (tested with 1.26.4)
3. **CREPE**: Upgraded to crepe>=0.0.16 (requires hmmlearn>=0.3.0, compatible with Python 3.11). The dependency is now included in pyproject.toml.
4. **Protobuf + note_seq**: Set `PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python` before running ddsp commands:
   ```bash
   export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
   ddsp_run --mode=train ...
   ```
5. **TensorFlow/Keras version compatibility**: DDSP requires TensorFlow 2.15.0 and Keras 2.15.0. Newer versions (2.20+) are incompatible due to Keras 3.x API changes.
6. **TFRecord file path**: Use absolute paths for TFRecord files, e.g., `/path/to/tfrecords/*.tfrecord*`
7. **Keras 3.x incompatibility**: Standalone Keras 3.x breaks tf.keras layers. Use TF 2.15.0 bundled Keras.
8. **Package Management**: Project uses `pyproject.toml` (PEP 621). Use `uv` for best experience or `pip install -e ".[extras]"` with pip.
9. **GPU Support**: A special activation script is provided to enable GPU support. It sets up CUDA libraries from the existing conda `ddsp_env`:
   ```bash
   source ./activate_gpu.sh
   # Or manually:
   export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
   export LD_LIBRARY_PATH=/home/jovyan/.uconda/envs/ddsp_env/lib:/usr/local/nvidia/lib:/usr/local/nvidia/lib64:$LD_LIBRARY_PATH
   ```

## Code to Avoid
- `absl.flags` (conflicts with pytest)
- `pkg_resources` (removed in setuptools 70+, use `os.path` instead)
- Deprecated TensorFlow APIs
- `from X import *` imports
- `setup.py` (project uses `pyproject.toml`, see PEP 621)