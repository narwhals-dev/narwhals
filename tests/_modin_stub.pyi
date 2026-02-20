"""Minimal stub for `tests/implementation_test.py::test_modin_typing`.

Used when `modin` isn't installed in the environment.
"""  # noqa: PYI021

from typing import Protocol

from narwhals._native import _ModinDataFrame, _ModinSeries

class Series(_ModinSeries, Protocol): ...

class DataFrame(_ModinDataFrame, Protocol):
    def duplicated(self) -> Series: ...

__all__ = ["DataFrame", "Series"]
