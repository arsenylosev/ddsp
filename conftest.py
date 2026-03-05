"""Pytest configuration for DDSP tests."""
import pytest


def pytest_configure(config):
    """Configure pytest for DDSP tests."""
    # Add markers for different test types
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "gpu: marks tests that require GPU"
    )


def pytest_collection_modifyitems(config, items):
    """Modify test collection."""
    # Skip tests that require crepe if not installed
    try:
        import crepe  # pylint: disable=unused-import
    except ImportError:
        skip_crepe = pytest.mark.skip(reason="crepe not installed")
        for item in items:
            if "crepe" in item.nodeid.lower():
                item.add_marker(skip_crepe)
