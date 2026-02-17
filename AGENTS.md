# AGENTS.md - DDSP Development Guide

AI agent guidelines for the DDSP (Differentiable Digital Signal Processing) codebase. Apache-2.0 licensed.

## Environment Setup

**Python**: 3.11-3.12 required (TensorFlow 2.15.1 + TFP 0.22.0)

**Installation**:
```bash
uv sync --no-build-isolation  # required for legacy packages like crepe
source .venv/bin/activate
```

## Build/Lint/Test Commands

### Testing
```bash
PYTHONPATH=$(pwd):$PYTHONPATH pytest                      # all tests
PYTHONPATH=$(pwd):$PYTHONPATH pytest ddsp/core_test.py    # single file
PYTHONPATH=$(pwd):$PYTHONPATH pytest ddsp/core_test.py::test_midi_to_hz_is_accurate  # single test
PYTHONPATH=$(pwd):$PYTHONPATH pytest -v --tb=short        # verbose (avoids absl.flags conflict)
PYTHONPATH=$(pwd):$PYTHONPATH pytest --cov=ddsp           # with coverage
```

### Linting
```bash
pylint ddsp              # entire codebase
pylint ddsp/core.py      # specific file
pylint --errors-only ddsp  # errors only
```

### Type Checking
```bash
mypy ddsp
```

## Code Style Guidelines

### Copyright Header
Every file must include the Apache-2.0 license header:
```python
# Copyright 2026 The DDSP Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
```

### Imports
Order: standard library → third-party → local. Group by type.
```python
from collections import abc
from typing import Any, Dict, Optional, Sequence, Text, TypeVar

import gin
import numpy as np
import tensorflow.compat.v2 as tf

from ddsp import core
```
- Use `import tensorflow.compat.v2 as tf`
- Alias Keras layers: `tfkl = tf.keras.layers`
- Alias TFP distributions: `tfd = tfp.distributions`
- Avoid `from X import *`

### Type Hints
Use hints for function signatures. Common types: `Dict`, `List`, `Optional`, `Text`, `Sequence`, `Any`. Define TypeVars for generics.
```python
TensorDict = Dict[Text, tf.Tensor]
Number = TypeVar('Number', int, float, np.ndarray, tf.Tensor)
```

### Naming Conventions
- Variables/functions: `snake_case`
- Classes: `PascalCase`
- Constants: `UPPER_SNAKE_CASE`
- Private members: `_private_method`
- Descriptive names: `n_samples`, `f0_hz`, `audio`

### Function Design
- Google-style docstrings (purpose, args, returns)
- Single responsibility principle
- Return early for error cases

### Error Handling
Use explicit exceptions with informative messages:
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
- Use `@gin.register` for functions/classes used in DAG configs
- Use `@gin.configurable` for globally configurable functions
- Keep core library agnostic to gin where possible

### Test Files
- Naming: `*_test.py` or `test_*.py`
- Use `unittest.TestCase` or `tf.test.TestCase`
- Use `@parameterized.named_parameters` for parameterized tests
- Tests in same directory as implementation
- Mock external dependencies

## File Organization

```
ddsp/
  core.py         # Core DSP functions
  processors.py   # Base Processor classes
  synths.py       # Synthesizer processors
  effects.py      # Effect processors
  losses.py       # Loss functions
  spectral_ops.py # Spectral operations
  dags.py         # DAG utilities
  training/       # Training infrastructure
```

## Known Issues

1. **absl.flags conflict**: `prepare_tfrecord_lib_test.py` fails with pytest verbose flags. Run without `-v` or use `--tb=short`.

## Code to Avoid

- `absl.flags` (conflicts with pytest)
- `pkg_resources` (removed in setuptools 70+)
- Deprecated TensorFlow APIs

## Key Dependencies

- TensorFlow 2.15.1, NumPy 1.26.4, SciPy 1.10.1
- librosa 0.10.0, gin-config 0.5.0, TensorFlow Probability 0.22.0
- apache-beam 2.59.0, pydub, tqdm, crepe 0.0.16
- pylint 2.x (required for apache-beam compatibility)
- setuptools>=60.0.0,<70.0.0 (required for crepe)

## Tensor Operations

- Use `core.tf_float32()` to ensure float32 conversion
- Handle batch dimensions explicitly: check `len(shape)` before operations
- Use `tf.squeeze()`/`tf.expand_dims()` for shape manipulation
- Prefer `tf.signal.stft` over `librosa.stft` for differentiable ops
- Cast numpy arrays to tensors before TF operations: `tf.constant(arr, dtype=tf.float32)`

## Audio Conventions

- Sample rate: 16000 Hz (default for CREPE pitch detection)
- Frame sizes: 2048, 4096, or 8192 (power of 2)
- Overlap: 0.75 (75%) standard for STFT
- MIDI range: 0-127, f0_hz: 0-20000 Hz
- Audio tensors: `[batch, samples]` or `[samples]` (squeeze channel dim)

## Git Workflow

- Create feature branches for changes
- Run linting and tests before committing
- Use conventional commits: `feat:`, `fix:`, `docs:`, `refactor:`
- Push to remote regularly for backup