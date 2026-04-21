"""Narwhals pytest plugin - auto-parametrises fixtures.

NOTE: All imports from `narwhals.*` are deferred inside the hook functions so that
the entry-point module can be loaded by pytest without pulling in the narwhals package tree.

This is critical because entry-point plugins are loaded *before* `coveragepy` starts
coverage measurement; any narwhals module imported at that stage would have its
module-level code (class definitions, constants, etc.) executed outside the coverage tracer.
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


def _default_backend_ids() -> list[str]:
    """Resolve the default `--nw-backends` value for the current environment.

    Honours `NARWHALS_DEFAULT_BACKENDS` if set, otherwise restricts
    [`DEFAULT_BACKENDS`][] to backends whose libraries are importable.
    """
    if env := os.environ.get("NARWHALS_DEFAULT_BACKENDS"):  # pragma: no cover
        return env.split(",")
    from narwhals.testing.constructors import DEFAULT_CONSTRUCTORS, frame_constructor

    return [
        name
        for name, constructor in frame_constructor._registry.items()
        if constructor.is_available and name in DEFAULT_CONSTRUCTORS
    ]


def pytest_addoption(parser: pytest.Parser) -> None:
    from narwhals.testing.constructors import DEFAULT_BACKENDS

    group = parser.getgroup("narwhals", "narwhals-testing")
    defaults = ", ".join(f"'{c}'" for c in sorted(DEFAULT_BACKENDS))
    group.addoption(
        "--nw-backends",
        action="store",
        default=",".join(_default_backend_ids()),
        type=str,
        help=(
            "Comma-separated list of (data|lazy) frame backend constructors to"
            f"parametrise. Defaults to the installed subset of ({defaults})"
        ),
    )
    group.addoption(
        "--all-nw-backends",
        action="store_true",
        default=False,
        help=("Run tests against every installed CPU backend (overrides --nw-backends)."),
    )
    # Escape hatch for downstream test suites that ship their own backend plugin.
    # When set, this plugin still adds the CLI options but stops parametrising the fixtures.
    group.addoption(
        "--use-external-nw-backend",
        action="store_true",
        default=False,
        help=(
            "Skip narwhals-testing's parametrisation and let another plugin "
            "provide the `nw_*frame_constructor` fixtures."
        ),
    )


def _select_backends(config: pytest.Config) -> list[FrameConstructor]:  # pragma: no cover
    from narwhals.testing.constructors import available_cpu_backends, prepare_backends

    _all_cpu_exclusions = frozenset({"modin", "pyspark[connect]"})

    if config.getoption("all_nw_backends"):
        selected = prepare_backends(
            include=available_cpu_backends(), exclude=_all_cpu_exclusions
        )
    else:
        opt = cast("str", config.getoption("nw_backends"))
        names = [c for c in opt.split(",") if c]
        selected = prepare_backends(include=names)

    if _pandas_version() < _MIN_PANDAS_NULLABLE_VERSION:
        _pandas_nullables = {"pandas[nullable]", "pandas[pyarrow]"}
        selected = [c for c in selected if c.name not in _pandas_nullables]
    return selected


def pytest_generate_tests(metafunc: pytest.Metafunc) -> None:
    if metafunc.config.getoption("use_external_nw_backend"):  # pragma: no cover
        return

    fixturenames = set(metafunc.fixturenames)
    if not fixturenames & {
        "nw_frame",
        "nw_dataframe",
        "nw_lazyframe",
        "nw_pandas_like_frame",
    }:
        return

    selected = _select_backends(metafunc.config)

    if "nw_dataframe" in fixturenames:
        params = [c for c in selected if c.is_eager]
        ids = [c.name for c in params]
        metafunc.parametrize("nw_dataframe", params, ids=ids)
    elif "nw_lazyframe" in fixturenames:  # pragma: no cover
        params = [c for c in selected if not c.is_eager]
        ids = [c.name for c in params]
        metafunc.parametrize("nw_dataframe", params, ids=ids)
    elif "nw_frame" in fixturenames:
        metafunc.parametrize("nw_frame", selected, ids=[c.name for c in selected])
    elif "nw_pandas_like_frame" in fixturenames:
        params = [c for c in selected if c.is_eager and c.is_pandas_like]
        ids = [c.name for c in params]
        metafunc.parametrize("nw_pandas_like_frame", params, ids=ids)
    else:  # pragma: no cover
        ...
