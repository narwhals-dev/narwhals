"""Pytest plugin to run narwhals' own test suite against `pizzlies`.

Usage: `python packages/pizzlies/run_tests.py`, see the package README.

`--use-external-constructor` makes narwhals' `tests/conftest.py` skip its own
constructor parametrization; the hooks below then parametrize the `constructor`
and `constructor_eager` fixtures with a plain-dict constructor instead, and
`constructor` also with a lazy one (`dict[lazy]`).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pytest

if TYPE_CHECKING:
    from pizzlies._plugin.lazy import LazyPlan
    from pizzlies._plugin.typing import DictFrame

CONSTRUCTORS_TO_SKIP = ("constructor_pandas_like",)


def dict_constructor(obj: dict[str, Any]) -> DictFrame:
    return {name: list(values) for name, values in obj.items()}


def dict_lazy_constructor(obj: dict[str, Any]) -> LazyPlan:
    from pizzlies._plugin.lazy import FramePlan

    return FramePlan(dict_constructor(obj))


@pytest.hookimpl(tryfirst=True)
def pytest_generate_tests(metafunc: pytest.Metafunc) -> None:
    if "constructor_eager" in metafunc.fixturenames:
        metafunc.parametrize("constructor_eager", [dict_constructor], ids=["dict"])
    elif "constructor" in metafunc.fixturenames:
        metafunc.parametrize(
            "constructor",
            [dict_constructor, dict_lazy_constructor],
            ids=["dict", "dict[lazy]"],
        )
    for fixture in CONSTRUCTORS_TO_SKIP:
        if fixture in metafunc.fixturenames:
            metafunc.parametrize(fixture, [], ids=[])
