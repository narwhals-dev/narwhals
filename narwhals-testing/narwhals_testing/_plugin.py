"""narwhals-testing pytest plugin.

Registered via the ``pytest11`` entry point. Handles two concerns:

1. Adds the ``narwhals_testing`` package directory to ``sys.path`` so that
   distributed test files can resolve ``from tests.utils import ...``.
   This happens at import time because entry-point plugin modules are
   imported before conftest files are loaded.

2. Registers CLI options (``--use-external-constructor``, etc.) so they are
   available even when tests are run via ``--pyargs``.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import pytest

_PACKAGE_DIR = str(Path(__file__).resolve().parent)

if _PACKAGE_DIR not in sys.path:
    sys.path.insert(0, _PACKAGE_DIR)


def pytest_addoption(parser: pytest.Parser) -> None:
    group = parser.getgroup("narwhals-testing")
    group.addoption(
        "--runslow", action="store_true", default=False, help="run slow tests"
    )
    group.addoption(
        "--all-cpu-constructors",
        action="store_true",
        default=False,
        help="run tests with all cpu constructors",
    )
    group.addoption(
        "--use-external-constructor",
        action="store_true",
        default=False,
        help="run tests with external constructor",
    )
    group.addoption(
        "--constructors", action="store", default=None, type=str, help="libraries to test"
    )
