"""Minimal stub for `tests/implementation_test.py::test_modin_typing`.

Used when `modin` isn't installed in the environment.
"""

# ruff: noqa: PYI021, PYI002
from typing import Literal, Protocol

from narwhals._native import _ModinDataFrame, _ModinSeries

MYPY: Literal[False] = False
"""https://mypy.readthedocs.io/en/stable/common_issues.html#python-version-and-system-platform-checks"""

if MYPY:
    # NOTE: `mypy` already ignores modin
    from modin.pandas import DataFrame, Series
else:
    try:
        import modin.pandas as _mpd

        Series = _mpd.Series
        DataFrame = _mpd.DataFrame
    except ImportError:
        class Series(_ModinSeries, Protocol): ...

        class DataFrame(_ModinDataFrame, Protocol):
            def duplicated(self) -> Series: ...

__all__ = ["DataFrame", "Series"]
