# AGENTS.md - DDSP Development Guide

AI agent guidelines for the DDSP (Differentiable Digital Signal Processing) codebase.

## Environment Setup

**Python**: 3.10 (TensorFlow 2.15 requires Python 3.10, tested with 3.10.19)

**Installation**:
```bash
conda create -n ddsp_env python=3.10.19 -y
conda activate ddsp_env
conda install -c conda-forge cudnn=8.9 cuda-toolkit=12.5 -y
pip install tensorflow==2.15.0 tensorflow-probability==0.22.0
pip install -e .[test,data_preparation]
```

**Console Scripts**: `ddsp_export`, `ddsp_run`, `ddsp_prepare_tfrecord`, `ddsp_generate_synthetic_dataset`

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
pytest                                  # all tests
pytest ddsp/core_test.py               # single file
pytest ddsp/core_test.py::test_midi_to_hz_is_accurate  # single test
PYTHONPATH=$(pwd):$PYTHONPATH pytest ddsp/spectral_ops_test.py
pytest -k "test_name" ddsp/            # filter tests by name pattern
pytest -x ddsp/                         # stop on first failure
pytest --tb=short ddsp/                 # shorter traceback
```

### Linting
```bash
pylint ddsp                            # entire codebase
pylint --errors-only ddsp              # errors only
pylint ddsp/core.py                    # single file
```

### Type Checking
```bash
mypy ddsp
mypy ddsp/core.py --ignore-missing-imports
```

### Running ddsp_run
```bash
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python && ddsp_run \
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
3. **CREPE build**: Requires `setuptools>=60.0.0,<70.0.0` and `--no-build-isolation`.
4. **Protobuf + note_seq**: Set `PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python` before running ddsp commands.
5. **TensorFlow/Keras version compatibility**: DDSP requires TensorFlow 2.15.0 and Keras 2.15.0. Newer versions (2.20+) are incompatible due to Keras 3.x API changes.
6. **TFRecord file path**: Use absolute paths for TFRecord files, e.g., `/path/to/tfrecords/*.tfrecord*`
7. **Keras 3.x incompatibility**: Standalone Keras 3.x breaks tf.keras layers. Use TF 2.15.0 bundled Keras.

## Code to Avoid
- `absl.flags` (conflicts with pytest)
- `pkg_resources` (removed in setuptools 70+, use `os.path` instead)
- Deprecated TensorFlow APIs
- `from X import *` imports