"""Narwhals pytest plugin - auto-parametrises `constructor*` fixtures.

NOTE: All imports from `narwhals.*` are deferred inside the hook functions so
that the entry-point module can be loaded by pytest without pulling in the
narwhals package tree.

This is critical because entry-point plugins are loaded *before* `pytest-cov`
starts coverage measurement; any narwhals module imported at that stage would
have its module-level code (class definitions, constants, etc.) executed outside
the coverage tracer.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING, cast

if TYPE_CHECKING:
    import pytest

    from narwhals.testing.typing import FrameConstructor


_MIN_PANDAS_NULLABLE_VERSION: tuple[int, ...] = (2, 0, 0)
"""`pandas.convert_dtypes(dtype_backend=...)` requires pandas >= 2.0.0."""


def _pandas_version() -> tuple[int, ...]:
    try:
        import pandas as pd
    except ImportError:  # pragma: no cover
        return (0, 0, 0)

    from narwhals._utils import parse_version

    return parse_version(pd.__version__)


def _default_constructor_ids() -> list[str]:
    """Resolve the default `--constructors` value for the current environment.

    Honours `NARWHALS_DEFAULT_CONSTRUCTORS` if set, otherwise restricts
    [`DEFAULT_CONSTRUCTORS`][] to backends whose libraries are importable.
    """
    if env := os.environ.get("NARWHALS_DEFAULT_CONSTRUCTORS"):  # pragma: no cover
        return env.split(",")
    from narwhals.testing.constructors import DEFAULT_CONSTRUCTORS, prepare_constructors

    return [c.name for c in prepare_constructors(include=DEFAULT_CONSTRUCTORS)]


def pytest_addoption(parser: pytest.Parser) -> None:
    from narwhals.testing.constructors import DEFAULT_CONSTRUCTORS

    group = parser.getgroup("narwhals", "narwhals.testing")
    defaults = ", ".join(f"'{c}'" for c in sorted(DEFAULT_CONSTRUCTORS))
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
) -> list[FrameConstructor]:  # pragma: no cover
    from narwhals.testing.constructors import (
        available_cpu_constructors,
        prepare_constructors,
    )

    _all_cpu_exclusions = frozenset({"modin", "pyspark[connect]"})

    if config.getoption("all_cpu_constructors"):
        selected = prepare_constructors(
            include=available_cpu_constructors(), exclude=_all_cpu_exclusions
        )
    else:
        opt = cast("str", config.getoption("constructors"))
        names = [c for c in opt.split(",") if c]
        selected = prepare_constructors(include=names)

    if _pandas_version() < _MIN_PANDAS_NULLABLE_VERSION:
        _pandas_nullables = {"pandas[nullable]", "pandas[pyarrow]"}
        selected = [c for c in selected if c.name not in _pandas_nullables]
    return selected


def pytest_generate_tests(metafunc: pytest.Metafunc) -> None:
    if metafunc.config.getoption("use_external_constructor"):  # pragma: no cover
        return

    fixturenames = set(metafunc.fixturenames)
    if not fixturenames & {"constructor", "constructor_eager", "constructor_pandas_like"}:
        return

    selected = _select_constructors(metafunc.config)

    if "constructor_eager" in fixturenames:
        params = [c for c in selected if c.is_eager]
        ids = [c.name for c in params]
        metafunc.parametrize("constructor_eager", params, ids=ids)
    elif "constructor" in fixturenames:
        metafunc.parametrize("constructor", selected, ids=[c.name for c in selected])
    elif "constructor_pandas_like" in fixturenames:
        params = [c for c in selected if c.is_eager and c.is_pandas_like]
        ids = [c.name for c in params]
        metafunc.parametrize("constructor_pandas_like", params, ids=ids)
    else:  # pragma: no cover
        ...
