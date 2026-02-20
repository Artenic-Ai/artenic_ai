"""Smoke tests for artenic_optimizer."""

from __future__ import annotations


def test_version() -> None:
    import artenic_optimizer

    assert artenic_optimizer.__version__ == "0.7.0"


def test_importable() -> None:
    import artenic_optimizer

    assert hasattr(artenic_optimizer, "__version__")
