from __future__ import annotations

from dataclasses import asdict
from dataclasses import dataclass
from typing import Sequence

import pandas as pd
import polars as pl

import narwhals.stable.v1 as nw


def test_dataclass() -> None:
    # https://github.com/narwhals-dev/narwhals/issues/1486
    @dataclass
    class Foo:
        a: Sequence[int]

    # dry-run to check that none of these error
    asdict(Foo(pd.Series([1, 2, 3])))  # type: ignore[arg-type]
    asdict(Foo(pl.Series([1, 2, 3])))  # type: ignore[arg-type]
    asdict(Foo(nw.from_native(pl.Series([1, 2, 3]), series_only=True)))  # type: ignore[arg-type]
    asdict(Foo(nw.from_native(pd.Series([1, 2, 3]), series_only=True)))  # type: ignore[arg-type]
