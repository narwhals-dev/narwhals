from __future__ import annotations

import os
from typing import TYPE_CHECKING, cast

from narwhals._utils import parse_version
from narwhals.testing.constructors import (
    ALL_CPU_CONSTRUCTORS,
    DEFAULT_CONSTRUCTORS,
    ConstructorBase,
    ConstructorName,
    prepare_constructors,
)

if TYPE_CHECKING:
    import pytest


_MIN_PANDAS_NULLABLE_VERSION: tuple[int, ...] = (2, 0, 0)
"""`pandas.convert_dtypes(dtype_backend=...)` requires pandas >= 2.0.0."""

_PANDAS_NULLABLES = {ConstructorName.PANDAS_NULLABLE, ConstructorName.PANDAS_PYARROW}

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
    return [str(c.name) for c in prepare_constructors(include=DEFAULT_CONSTRUCTORS)]


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
) -> list[ConstructorBase]:  # pragma: no cover
    if config.getoption("all_cpu_constructors"):
        selected = prepare_constructors(
            include=ALL_CPU_CONSTRUCTORS, exclude=_ALL_CPU_EXCLUSIONS
        )
    else:
        opt = cast("str", config.getoption("constructors"))
        names = [ConstructorName(c) for c in opt.split(",") if c]
        selected = prepare_constructors(include=names)

    if _pandas_version() < _MIN_PANDAS_NULLABLE_VERSION:
        selected = [c for c in selected if c.name not in _PANDAS_NULLABLES]
    return selected


def pytest_generate_tests(metafunc: pytest.Metafunc) -> None:
    if metafunc.config.getoption("use_external_constructor"):  # pragma: no cover
        return

    fixturenames = set(metafunc.fixturenames)
    if not fixturenames & {"constructor", "constructor_eager", "constructor_pandas_like"}:
        return

    selected = _select_constructors(metafunc.config)

    if "constructor_eager" in fixturenames:
        params = [c for c in selected if c.name.is_eager]
        ids = [str(c.name) for c in params]
        metafunc.parametrize("constructor_eager", params, ids=ids)
    elif "constructor" in fixturenames:
        metafunc.parametrize("constructor", selected, ids=[str(c.name) for c in selected])
    elif "constructor_pandas_like" in fixturenames:
        params = [c for c in selected if c.name.is_eager and c.name.is_pandas_like]
        ids = [str(c.name) for c in params]
        metafunc.parametrize("constructor_pandas_like", params, ids=ids)
