"""narwhals-testing: Narwhals test suite distribution for plugin authors.

Usage::

    pip install narwhals-testing my-narwhals-plugin
    pytest --pyargs narwhals_testing.tests --use-external-constructor
"""

from __future__ import annotations

from pathlib import Path

__version__ = "0.1.0"


def get_test_dir() -> Path:
    """Return the path to the distributed tests directory."""
    return Path(__file__).resolve().parent / "tests"
