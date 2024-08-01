"""
Dask support in Narwhals is still _very_ scant.

Start with a simple test file whilst we develop the basics.
Once we're a bit further along (say, we can at least evaluate
TPC-H Q1 with Dask), then we can integrate dask tests into
the main test suite.
"""

from __future__ import annotations

import sys
import warnings
from datetime import datetime

import pandas as pd
import pytest

import narwhals.stable.v1 as nw
from tests.utils import compare_dicts

pytest.importorskip("dask")
with warnings.catch_warnings():
    warnings.filterwarnings(
        "ignore",
        category=pytest.PytestDeprecationWarning,
    )
    pytest.importorskip("dask_expr")


if sys.version_info < (3, 9):
    pytest.skip("Dask tests require Python 3.9+", allow_module_level=True)


def test_to_datetime() -> None:
    import dask.dataframe as dd

    data = {"a": ["2020-01-01T12:34:56"]}
    dfdd = dd.from_pandas(pd.DataFrame(data))
    df = nw.from_native(dfdd)

    format = "%Y-%m-%dT%H:%M:%S"
    result = df.with_columns(b=nw.col("a").str.to_datetime(format=format))

    expected = {
        "a": ["2020-01-01T12:34:56"],
        "b": [datetime.strptime("2020-01-01T12:34:56", format)],  # noqa: DTZ007
    }
    compare_dicts(result, expected)
