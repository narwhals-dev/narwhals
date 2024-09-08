from __future__ import annotations

from typing import TYPE_CHECKING
from typing import Any

import numpy as np
import pytest

import narwhals.stable.v1 as nw
from tests.utils import compare_dicts

if TYPE_CHECKING:
    from narwhals.dtypes import DType


@pytest.mark.parametrize(
    ("dtype", "expected_lit"),
    [(None, [2, 2, 2]), (nw.String, ["2", "2", "2"]), (nw.Float32, [2.0, 2.0, 2.0])],
)
def test_lit(
    constructor: Any,
    dtype: DType | None,
    expected_lit: list[Any],
    request: pytest.FixtureRequest,
) -> None:
    if "dask" in str(constructor) and dtype == nw.String:
        request.applymarker(pytest.mark.xfail)
    if "pyspark" in str(constructor):
        request.applymarker(pytest.mark.xfail)
    data = {"a": [1, 3, 2], "b": [4, 4, 6], "z": [7.0, 8, 9]}
    df_raw = constructor(data)
    df = nw.from_native(df_raw).lazy()
    result = df.with_columns(nw.lit(2, dtype).alias("lit"))
    expected = {
        "a": [1, 3, 2],
        "b": [4, 4, 6],
        "z": [7.0, 8.0, 9.0],
        "lit": expected_lit,
    }
    compare_dicts(result, expected)


def test_lit_error(request: pytest.FixtureRequest, constructor: Any) -> None:
    if "pyspark" in str(constructor):
        request.applymarker(pytest.mark.xfail)
    data = {"a": [1, 3, 2], "b": [4, 4, 6], "z": [7.0, 8, 9]}
    df_raw = constructor(data)
    df = nw.from_native(df_raw).lazy()
    with pytest.raises(
        ValueError, match="numpy arrays are not supported as literal values"
    ):
        _ = df.with_columns(nw.lit(np.array([1, 2])).alias("lit"))
    with pytest.raises(
        NotImplementedError, match="Nested datatypes are not supported yet."
    ):
        _ = df.with_columns(nw.lit((1, 2)).alias("lit"))
    with pytest.raises(
        NotImplementedError, match="Nested datatypes are not supported yet."
    ):
        _ = df.with_columns(nw.lit([1, 2]).alias("lit"))
