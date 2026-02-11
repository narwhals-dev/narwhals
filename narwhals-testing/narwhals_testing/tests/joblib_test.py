from __future__ import annotations

from typing import Any

import pytest

import narwhals as nw
from tests.utils import ConstructorEager, assert_equal_data

pytest.importorskip("joblib")

from joblib import Parallel, delayed


def test_parallelisability(constructor_eager: ConstructorEager) -> None:
    # https://github.com/narwhals-dev/narwhals/issues/2450
    def do_something(df: nw.DataFrame[Any]) -> nw.DataFrame[Any]:  # pragma: no cover
        return df.with_columns(nw.col("col1") * 2)

    dfs = [nw.from_native(constructor_eager({"col1": [0, 2], "col2": [3, 7]}))]
    result = list(Parallel(n_jobs=-1)(delayed(do_something)(df_) for df_ in dfs))
    assert len(result) == 1
    expected = {"col1": [0, 4], "col2": [3, 7]}
    assert_equal_data(result[0], expected)


def test_parallelisability_series(
    constructor_eager: ConstructorEager, request: pytest.FixtureRequest
) -> None:
    # https://github.com/narwhals-dev/narwhals/issues/2450
    if "modin" in str(constructor_eager):
        request.applymarker(pytest.mark.xfail)

    def do_something(s: nw.Series[Any]) -> nw.Series[Any]:  # pragma: no cover
        return s * 2

    columns = [
        nw.from_native(
            constructor_eager({"col1": [0, 2], "col2": [3, 7]}), eager_only=True
        )["col1"]
    ]
    result = list(
        Parallel(n_jobs=-1)(delayed(do_something)(column) for column in columns)
    )
    assert len(result) == 1
    assert_equal_data({"col1": result[0]}, {"col1": [0, 4]})
