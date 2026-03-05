# AGENTS.md - DDSP Development Guide

AI agent guidelines for the DDSP (Differentiable Digital Signal Processing) codebase.

## Environment Setup

**Prerequisites**: Python 3.10-3.11, NVIDIA driver ≥525 (for GPU)

**One-Command Setup**:
```bash
uv sync --extra data_preparation --extra test
```

**CREPE Workaround** (required if sync fails):
```bash
uv pip install "setuptools<70" wheel && uv pip install crepe>=0.0.16 --no-build-isolation && rm -rf .venv && uv sync --extra data_preparation --extra test
```

## Build/Lint/Test Commands

### Testing
```bash
# Single test (correct format with class prefix)
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python && uv run pytest ddsp/core_test.py::UtilitiesTest::test_midi_to_hz_is_accurate

# Single file
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python && uv run pytest ddsp/core_test.py

# All tests
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python && uv run pytest

# Filter by name pattern
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python && uv run pytest -k "test_name" ddsp/

# CI test command (no verbose flags)
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python && uv run pytest --tb=short
```

### Linting & Formatting
```bash
# Check all files
uv run ruff check ddsp

# Format all files
uv run ruff format ddsp

# Check single file
uv run ruff check ddsp/core.py

# CI format check (dry-run)
uv run ruff format --check ddsp
```

### Running ddsp_run
```bash
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python && uv run ddsp_run \
  --mode=train \
  --save_dir=/path/to/experiments \
  --gin_file=ddsp/training/gin/models/solo_instrument.gin \
  --gin_file=ddsp/training/gin/datasets/tfrecord.gin \
  --gin_param="TFRecordProvider.file_pattern='/absolute/path/to/*.tfrecord*'"
```

## Code Style Guidelines

### Copyright Header
```python
# Copyright 2026 The DDSP Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
```

### Imports
Order: stdlib → third-party → local. Group with blank lines.
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
- Aliases: `tfkl = tf.keras.layers`, `tfd = tfp.distributions`
- Avoid `from X import *`

### Formatting
- **Line length**: 80 characters max
- **Indentation**: 2 spaces (4 after paren continuation)
- **Quotes**: Single quotes preferred
- Docstrings for all public functions

### Naming Conventions
- Variables/functions: `snake_case`
- Classes: `PascalCase`
- Constants: `UPPER_SNAKE_CASE`
- Private: `_private_method`

### Type Hints & Function Design
```python
TensorDict = Dict[Text, tf.Tensor]
Number = TypeVar('Number', int, float, np.ndarray, tf.Tensor)
```
- Google-style docstrings (Args, Returns sections)
- Return early for error cases
- Raise errors with descriptive messages:
```python
raise ValueError(f'Keys: {keys} must be same length as {x}')
raise NotImplementedError  # for abstract methods
```

### TensorFlow Conventions
- Use `tf.float32` explicitly (not `tf.float16`)
- Check `tf.executing_eagerly()` for eager/graph decisions
- Use `tf.function` for performance-critical code
- Prefer Keras layers (`tfkl`), set `autocast=False`

### Gin Configuration
- Use `@gin.register` for DAG-configurable components
- Use `@gin.configurable` for global configuration
- Gin files in `ddsp/training/gin/`

### Test Files
- Naming: `*_test.py`, base: `parameterized.TestCase, tf.test.TestCase`
- Place in same directory, classes: `Test*`, methods: `test_*`
- **Avoid `absl.flags` in test files** - it conflicts with pytest CLI flags
- Use `tempfile.mkdtemp()` for temp directories, not `absl.flags`

## Project Structure
```
ddsp/
  core.py,spectral_ops.py,processors.py,synths.py,effects.py
  training/ddsp_run.py,gin/,models/,decoders/,data_preparation/
```

## Audio Conventions
- Sample rate: 16000 Hz (default)
- Frame sizes: 2048, 4096, 8192 (power of 2)
- CREPE produces frames at 100 fps (hop_size=160 at 16kHz)
- Feature frame rates (e.g., 250 fps) may differ from audio sample rate
- **Important**: When resampling audio, handle audio_16k and audio at different sample rates separately
- Overlap: 0.75 (75%) standard for STFT
- MIDI range: 0-127, f0_hz: 0-20000 Hz
- Audio tensors: `[batch, samples]` or `[samples]`

## Known Issues & Fixes

1. **absl.flags**: Fails with pytest `-v` - use `tempfile.mkdtemp()` for temp dirs instead
2. **NumPy**: Use `numpy<2` (1.26.4 tested)
3. **Protobuf**: Set `PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python` and pin `protobuf<4`
4. **TFRecord paths**: Use absolute paths
5. **TensorFlow**: Requires TF 2.15.0 (incompatible with TF 2.20+)
6. **CREPE**: Pre-install `setuptools<70` with `--no-build-isolation`
7. **CREPE frame rate mismatch**: CREPE outputs at 100 fps but code expects configurable `frame_rate`. Resample f0_hz/f0_confidence using `np.interp()` when frame rates differ.
8. **Splitting audio/features**: When splitting examples, calculate window counts separately for audio (sample_rate), audio_16k (16000 Hz), and features (frame_rate) - they have different lengths even for same duration.

## Code to Avoid
- `absl.flags` (conflicts with pytest)
- `pkg_resources` (removed in setuptools 70+)
- Deprecated TensorFlow APIs
- `from X import *` imports