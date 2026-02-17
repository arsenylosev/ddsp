# AGENTS.md - DDSP Development Guide

AI agent guidelines for working on the DDSP (Differentiable Digital Signal Processing) codebase. Version 3.7.0, Apache-2.0 licensed.

## Environment Setup

**Python**: 3.10 required (TensorFlow 2.11 incompatibility with 3.11)

**Installation**:
```bash
uv sync --no-build-isolation  # recommended (required for legacy packages like crepe)
pip install -e .[data_preparation,test]  # alternative
```

**Dependencies**: `pyproject.toml` and `requirements.txt`

## Build/Lint/Test Commands

### Testing
```bash
pytest                      # all tests
pytest ddsp/core_test.py    # single file
pytest ddsp/core_test.py::test_midi_to_hz_is_accurate  # single test
pytest -v                   # verbose
pytest --cov=ddsp           # with coverage
```

### Linting
```bash
pylint ddsp           # entire codebase
pylint ddsp/core.py   # specific file
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
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
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

### Naming
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

## Data Preparation Optimizations

### ddsp_prepare_tfrecord Performance

The `ddsp_prepare_tfrecord` command has been optimized with the following improvements:

**Single-Pass Audio Loading**: Audio is loaded once at 16kHz and resampled if needed, eliminating redundant I/O (~50% reduction for non-16kHz).

**Combined Feature Computation**: F0 and loudness are computed in a single `beam.Map` call.

**Progress Tracking**: Thread-safe ProgressTracker with `--progress_bar`, `--max_workers`, `--cache_size` flags. Uses logging-based output compatible with Apache Beam parallel processing.

## Known Issues

1. **Test failures**: `absl.flags` conflicts with pytest arguments in `prepare_tfrecord_lib_test.py`
2. **Python version**: Must use 3.10, not 3.11

## Key Dependencies

- TensorFlow 2.11.0, NumPy 1.23.5, SciPy 1.10.1
- librosa 0.10.0, gin-config 0.5.0, TensorFlow Probability 0.19.0
- apache-beam 2.46.0, pydub, tqdm, crepe 0.0.16

## Code to Avoid

- `absl.flags` (conflicts with pytest)
- `pkg_resources` (removed in setuptools 70+)
- Deprecated TensorFlow APIs