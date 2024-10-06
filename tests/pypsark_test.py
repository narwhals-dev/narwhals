"""
PySpark support in Narwhals is still _very_ limited.
Start with a simple test file whilst we develop the basics.
Once we're a bit further along, we can integrate PySpark tests into the main test suite.
"""

from __future__ import annotations

from typing import TYPE_CHECKING
from typing import Any

import pandas as pd

import narwhals.stable.v1 as nw

if TYPE_CHECKING:
    from pyspark.sql import SparkSession

    from narwhals.typing import IntoFrame


def pyspark_constructor(obj: Any, spark_session: SparkSession) -> IntoFrame:
    return spark_session.createDataFrame(pd.DataFrame(obj))  # type: ignore[no-any-return]


def test_columns(spark_session: SparkSession) -> None:
    data = {"a": [1, 3, 2], "b": [4, 4, 6], "z": [7.0, 8, 9]}
    df = nw.from_native(pyspark_constructor(data, spark_session))
    result = df.columns
    expected = ["a", "b", "z"]
    assert result == expected


def test_with_columns() -> None:
    raise NotImplementedError
