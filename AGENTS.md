# AGENTS.md - DDSP Development Guide

AI agent guidelines for working on the DDSP (Differentiable Digital Signal Processing) codebase. Version 3.7.0, Apache-2.0 licensed.

## Environment Setup

**Python**: 3.10 required (TensorFlow 2.11 incompatibility with 3.11)

**Installation**:
```bash
uv sync  # recommended
pip install -e .[data_preparation,test]  # alternative
```

**Dependencies**: `pyproject.toml` and `requirements.txt`

## Build/Lint/Test Commands

### Testing
```bash
pytest                      # all tests
pytest ddsp/core_test.py    # single file
pytest ddsp/core_test.py::test_foo_function  # single test
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
Every file must include the Apache-2.0 license header.

### Imports
Order: standard library → third-party → local. Group by type.
```python
from collections import abc
from typing import Any, Dict, Optional, Sequence, Text, TypeVar
import numpy as np
import tensorflow.compat.v2 as tf
from ddsp import core
```
- Use `import tensorflow.compat.v2 as tf`
- Alias Keras layers: `tfkl = tf.keras.layers`
- Avoid `from X import *`

### Type Hints
Use hints for function signatures. Common types: `Dict`, `List`, `Optional`, `Text`, `Sequence`, `Any`. Define TypeVars for generics.

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
raise ValueError(f'Message: {detail}')
raise NotImplementedError  # for abstract methods
```

### TensorFlow Conventions
- Use `tf.float32` explicitly
- Check `tf.executing_eagerly()` for eager/graph decisions
- Use `tf.function` for performance-critical code
- Prefer Keras layers (`tfkl`)
- Set `autocast=False` in custom layers

### Configuration with Gin
Use `@gin.configurable` decorator for runtime-configurable classes/functions. Keep core library agnostic to gin where possible.

### Test Files
- Naming: `*_test.py` or `test_*.py`
- Use `unittest` assertions or pytest
- Tests in same directory as implementation
- Mock external dependencies

### File Organization
```
ddsp/
  core.py         # Core DSP functions
  processors.py   # Base Processor classes
  synths.py       # Synthesizer processors
  effects.py      # Effect processors
  losses.py       # Loss functions
  spectral_ops.py # Spectral operations
  training/       # Training infrastructure
```

### Code to Avoid
- `absl.flags` (conflicts with pytest)
- `pkg_resources` (removed in setuptools 70+)
- Deprecated TensorFlow APIs

## Known Issues

1. **12 test failures** in `prepare_tfrecord_lib_test.py`: `absl.flags` conflicts with pytest arguments (infrastructure issue)
2. **Python version**: Must use 3.10, not 3.11

## Key Dependencies
- TensorFlow 2.11.0, NumPy 1.23.5, SciPy 1.10.1
- librosa 0.10.0, gin-config 0.5.0, TensorFlow Probability 0.19.0