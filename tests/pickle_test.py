from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING

import pytest

import narwhals as nw

if TYPE_CHECKING:
    from collections.abc import Sequence


# https://github.com/narwhals-dev/narwhals/issues/1486
@dataclass
class Foo:
    a: Sequence[int]


def test_dataclass_pandas() -> None:
    pytest.importorskip("pandas")
    import pandas as pd

    # dry-run to check that none of these error
    asdict(Foo(pd.Series([1, 2, 3])))  # type: ignore[arg-type]
    asdict(Foo(nw.from_native(pd.Series([1, 2, 3]), series_only=True)))  # type: ignore[arg-type]


def test_dataclass_polars() -> None:
    pytest.importorskip("polars")
    import polars as pl

    # dry-run to check that none of these error
    asdict(Foo(pl.Series([1, 2, 3])))  # type: ignore[arg-type]
    asdict(Foo(nw.from_native(pl.Series([1, 2, 3]), series_only=True)))  # type: ignore[arg-type]
