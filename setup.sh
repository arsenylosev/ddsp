#!/bin/bash
# DDSP One-Command Setup Script
# Usage: curl -fsSL https://raw.githubusercontent.com/magenta/ddsp/main/setup.sh | bash
# Or run locally: bash setup.sh

set -e

echo "=== DDSP Environment Setup ==="
echo

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if we're in the ddsp repository
if [ ! -f "pyproject.toml" ]; then
    print_error "Not in DDSP repository root directory"
    print_status "Please run this script from the DDSP repository root"
    exit 1
fi

# Check if pyproject.toml contains ddsp
if ! grep -q "name = \"ddsp\"" pyproject.toml; then
    print_error "pyproject.toml doesn't appear to be from DDSP"
    exit 1
fi

print_status "Found DDSP repository"

# Check for uv
if ! command -v uv &> /dev/null; then
    print_status "uv not found, installing..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    # Add to PATH for current session
    export PATH="$HOME/.local/bin:$PATH"
    
    if ! command -v uv &> /dev/null; then
        print_error "Failed to install uv. Please install manually:"
        print_error "  curl -LsSf https://astral.sh/uv/install.sh | sh"
        exit 1
    fi
    print_status "uv installed successfully"
else
    print_status "uv found: $(uv --version)"
fi

# Check Python version (need 3.10-3.11)
print_status "Checking Python version..."
python_version=$(python3 --version 2>&1 | grep -oP '\d+\.\d+' | head -1)
major=$(echo $python_version | cut -d. -f1)
minor=$(echo $python_version | cut -d. -f2)

if [ "$major" -eq 3 ] && [ "$minor" -ge 10 ] && [ "$minor" -le 11 ]; then
    print_status "Python $python_version is compatible"
else
    print_warning "Python $python_version detected (need 3.10-3.11)"
    print_status "uv will use appropriate Python version for the virtual environment"
fi

# Check NVIDIA driver for GPU support
print_status "Checking NVIDIA driver..."
if command -v nvidia-smi &> /dev/null; then
    driver_version=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -1 || echo "")
    if [ -n "$driver_version" ]; then
        print_status "NVIDIA driver detected: $driver_version"
        # Check if driver is >= 525
        driver_major=$(echo $driver_version | cut -d. -f1)
        if [ "$driver_major" -ge 525 ]; then
            print_status "Driver version is compatible (>= 525)"
        else
            print_warning "Driver version $driver_version may be too old (recommend >= 525)"
        fi
    else
        print_warning "nvidia-smi found but couldn't detect driver version"
    fi
else
    print_warning "nvidia-smi not found. GPU support may not be available."
    print_status "To use GPU, please install NVIDIA drivers (>= 525)"
fi

# Remove existing virtual environment if it exists
if [ -d ".venv" ]; then
    print_status "Removing existing .venv..."
    rm -rf .venv
fi

# Run uv sync
print_status "Installing dependencies with uv sync..."
print_status "This may take a few minutes..."

if uv sync --extra data_preparation --extra test --no-build-isolation; then
    print_status "Dependencies installed successfully"
else
    print_error "Failed to install dependencies"
    print_status "Trying with crepe pre-installation..."
    
    # Try the workaround with setuptools
    uv venv --python 3.11
    source .venv/bin/activate
    uv pip install "setuptools<70" wheel hatchling editables
    uv pip install crepe==0.0.16 --no-build-isolation
    
    if uv sync --extra data_preparation --extra test --no-build-isolation; then
        print_status "Dependencies installed successfully (with workaround)"
    else
        print_error "Failed to install dependencies even with workaround"
        exit 1
    fi
fi

print_status "Environment setup complete!"
echo

# Verify installation
print_status "Verifying installation..."
echo

# Activate and run verification
echo "=== Testing Python imports ==="
if source .venv/bin/activate && \
   python -c "import ddsp; print(f'DDSP version: {ddsp.__version__}')" && \
   python -c "import tensorflow as tf; print(f'TensorFlow version: {tf.__version__}')"; then
    print_status "Python imports successful"
else
    print_error "Python import test failed"
    exit 1
fi

echo
echo "=== Testing GPU availability ==="
python -c "
import tensorflow as tf
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    print(f'✓ GPU available: {len(gpus)} device(s)')
    for gpu in gpus:
        print(f'  - {gpu}')
else:
    print('✗ No GPU detected (CPU only)')
    print('  Run: source activate_gpu.sh')
    print('  Then test again: python -c \"import tensorflow as tf; print(tf.config.list_physical_devices(\\\"GPU\\\"))\"')
"

echo
echo "=== Running quick test ==="
if uv run pytest ddsp/core_test.py::UtilitiesTest::test_midi_to_hz_is_accurate -q; then
    print_status "Quick test passed"
else
    print_warning "Quick test failed (non-critical)"
fi

echo
print_status "Setup complete!"
echo
echo "Next steps:"
echo "  1. Activate environment: source .venv/bin/activate"
echo "  2. For GPU support: source activate_gpu.sh"
echo "  3. Run tests: uv run pytest"
echo
echo "Or use uv run directly (no activation needed):"
echo "  uv run python your_script.py"
echo "  uv run pytest"
