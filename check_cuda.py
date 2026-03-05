#!/usr/bin/env python3
"""CUDA auto-detection and installation helper for DDSP."""

import subprocess
import sys


def check_cuda_available():
    """Check if CUDA-capable GPU is available."""
    try:
        # Try importing tensorflow first
        import tensorflow as tf
        gpus = tf.config.list_physical_devices('GPU')
        return len(gpus) > 0
    except ImportError:
        pass
    
    # Fallback: check nvidia-smi
    try:
        result = subprocess.run(
            ['nvidia-smi'],
            capture_output=True,
            text=True,
            timeout=5
        )
        return result.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    
    # Fallback: check /proc/driver/nvidia
    try:
        import os
        return os.path.exists('/proc/driver/nvidia/gpus')
    except:
        pass
    
    return False


def get_install_command():
    """Get the appropriate install command based on GPU availability."""
    if check_cuda_available():
        print("✓ CUDA GPU detected!")
        print("Install with: uv pip install -e '.[cuda]'")
        return ['uv', 'pip', 'install', '-e', '.[cuda]']
    else:
        print("✗ No CUDA GPU detected")
        print("Install with: uv pip install -e '.'")
        return ['uv', 'pip', 'install', '-e', '.']


def main():
    print("DDSP CUDA Auto-Detection")
    print("=" * 40)
    
    has_cuda = check_cuda_available()
    
    print(f"\nCUDA Available: {has_cuda}")
    print(f"\nRecommended install command:")
    cmd = get_install_command()
    print(f"  {' '.join(cmd)}")
    
    if '--install' in sys.argv:
        print("\nRunning installation...")
        subprocess.run(cmd, check=True)
        print("✓ Installation complete!")
    
    return 0 if has_cuda else 1


if __name__ == '__main__':
    sys.exit(main())
