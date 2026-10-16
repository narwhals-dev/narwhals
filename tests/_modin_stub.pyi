"""Minimal stub for `tests/implementation_test.py::test_modin_typing`.

`modin` isn't a typing dependency, so checkers that ignore missing imports would see
it as `Any`, which matches every `Implementation` overload. Use a protocol instead.
"""

# ruff: noqa: PYI021
from typing import Protocol

from narwhals._native import _ModinDataFrame, _ModinSeries

class Series(_ModinSeries, Protocol): ...

class DataFrame(_ModinDataFrame, Protocol):
    def duplicated(self) -> Series: ...

__all__ = ["DataFrame", "Series"]
