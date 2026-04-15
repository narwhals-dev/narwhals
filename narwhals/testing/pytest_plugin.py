from __future__ import annotations

import os
from typing import TYPE_CHECKING, cast

from narwhals._utils import parse_version
from narwhals.testing.constructors import (
    ALL_CPU_CONSTRUCTORS,
    DEFAULT_CONSTRUCTORS,
    ConstructorBase,
    ConstructorEagerBase,
    ConstructorName,
    available_constructors,
    get_constructor,
)

if TYPE_CHECKING:
    from collections.abc import Iterable

    import pytest


_MIN_PANDAS_NULLABLE_VERSION: tuple[int, ...] = (2, 0, 0)
"""`pandas.convert_dtypes(dtype_backend=...)` requires pandas >= 2.0.0."""

_ALL_CPU_EXCLUSIONS: frozenset[ConstructorName] = frozenset(
    {ConstructorName.MODIN, ConstructorName.PYSPARK_CONNECT}
)
"""Backends excluded from `--all-cpu-constructors` even when installed:

* modin is too slow for the full matrix
* pyspark[connect] needs a different local setup and can't run alongside pyspark
"""


def _pandas_version() -> tuple[int, ...]:
    try:
        import pandas as pd
    except ImportError:  # pragma: no cover
        return (0, 0, 0)
    return parse_version(pd.__version__)


def _default_constructor_ids() -> list[str]:
    """Resolve the default `--constructors` value for the current environment.

    Honours `NARWHALS_DEFAULT_CONSTRUCTORS` if set, otherwise restricts
    [`DEFAULT_CONSTRUCTORS`][] to backends whose libraries are importable.
    """
    if env := os.environ.get("NARWHALS_DEFAULT_CONSTRUCTORS"):  # pragma: no cover
        return env.split(",")
    available = available_constructors()
    return [name.value for name in DEFAULT_CONSTRUCTORS if name in available]


def pytest_addoption(parser: pytest.Parser) -> None:
    group = parser.getgroup("narwhals", "narwhals.testing")
    defaults = ", ".join(f"'{c.value}'" for c in sorted(DEFAULT_CONSTRUCTORS))
    group.addoption(
        "--constructors",
        action="store",
        default=",".join(_default_constructor_ids()),
        type=str,
        help=(
            "Comma-separated list of backend constructors to parametrise. "
            f"Defaults to the installed subset of ({defaults})"
        ),
    )
    group.addoption(
        "--all-cpu-constructors",
        action="store_true",
        default=False,
        help=(
            "Run tests against every installed CPU constructor "
            "(overrides --constructors)."
        ),
    )
    # Escape hatch for downstream test suites that ship their own constructor
    # plugin. When set, this plugin still adds the CLI options but stops
    # parametrising the fixtures.
    group.addoption(
        "--use-external-constructor",
        action="store_true",
        default=False,
        help=(
            "Skip narwhals.testing's parametrisation and let another plugin "
            "provide the `constructor*` fixtures."
        ),
    )


def _select_constructors(
    config: pytest.Config,
) -> list[ConstructorName]:  # pragma: no cover
    if config.getoption("all_cpu_constructors"):
        names: Iterable[ConstructorName] = sorted(
            ALL_CPU_CONSTRUCTORS - _ALL_CPU_EXCLUSIONS, key=lambda c: c.value
        )
    else:
        opt = cast("str", config.getoption("constructors"))
        names = [ConstructorName(c) for c in opt.split(",") if c]

    pandas_version = _pandas_version()
    selected: list[ConstructorName] = []
    for name in names:
        if (
            name in {ConstructorName.PANDAS_NULLABLE, ConstructorName.PANDAS_PYARROW}
            and pandas_version < _MIN_PANDAS_NULLABLE_VERSION
        ):
            continue
        selected.append(name)
    return selected


def pytest_generate_tests(metafunc: pytest.Metafunc) -> None:
    if metafunc.config.getoption("use_external_constructor"):  # pragma: no cover
        return

    fixturenames = set(metafunc.fixturenames)
    if not fixturenames & {"constructor", "constructor_eager", "constructor_pandas_like"}:
        return

    selected = _select_constructors(metafunc.config)

    constructors: list[ConstructorBase] = []
    constructor_ids: list[str] = []
    eager: list[ConstructorEagerBase] = []
    eager_ids: list[str] = []
    pandas_like: list[ConstructorEagerBase] = []
    pandas_like_ids: list[str] = []

    for name in selected:
        constructor = get_constructor(name)
        constructors.append(constructor)
        constructor_ids.append(name.value)
        if isinstance(constructor, ConstructorEagerBase):
            eager.append(constructor)
            eager_ids.append(name.value)
            if name.is_pandas_like:
                pandas_like.append(constructor)
                pandas_like_ids.append(name.value)

    if "constructor_eager" in fixturenames:
        metafunc.parametrize("constructor_eager", eager, ids=eager_ids)
    elif "constructor" in fixturenames:
        metafunc.parametrize("constructor", constructors, ids=constructor_ids)
    elif "constructor_pandas_like" in metafunc.fixturenames:
        metafunc.parametrize("constructor_pandas_like", pandas_like, ids=pandas_like_ids)
