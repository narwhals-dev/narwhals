"""Pytest plugin to run narwhals' own test suite against `narwhals-dict`.

Usage: `python packages/narwhals-dict/run_tests.py`, see the package README.

`--use-external-constructor` makes narwhals' `tests/conftest.py` skip its own
constructor parametrization; the hooks below then parametrize the `constructor`
and `constructor_eager` fixtures with a plain-dict constructor instead.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pytest

if TYPE_CHECKING:
    from narwhals_dict.typing import DictFrame

CONSTRUCTORS_TO_SKIP = ("constructor_pandas_like",)


def dict_constructor(obj: dict[str, Any]) -> DictFrame:
    return {name: list(values) for name, values in obj.items()}


@pytest.hookimpl(tryfirst=True)
def pytest_generate_tests(metafunc: pytest.Metafunc) -> None:
    if "constructor_eager" in metafunc.fixturenames:
        metafunc.parametrize("constructor_eager", [dict_constructor], ids=["dict"])
    elif "constructor" in metafunc.fixturenames:
        metafunc.parametrize("constructor", [dict_constructor], ids=["dict"])
    for fixture in CONSTRUCTORS_TO_SKIP:
        if fixture in metafunc.fixturenames:
            metafunc.parametrize(fixture, [], ids=[])
