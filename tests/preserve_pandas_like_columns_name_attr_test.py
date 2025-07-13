from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

import narwhals as nw

if TYPE_CHECKING:
    from tests.utils import Constructor


def test_ops_preserve_column_index_name(
    constructor: Constructor, request: pytest.FixtureRequest
) -> None:
    if not any(x in str(constructor) for x in ("pandas", "modin", "cudf", "dask")):
        pytest.skip(
            reason="Dataframe columns is a list and do not have a `name` like a pandas Index does"
        )
    if "dask" in str(constructor):
        # https://github.com/dask/dask/issues/11874
        request.applymarker(pytest.mark.xfail)

    data = {"a": [1, 3, 2], "b": [4, 4, 6], "z": [7.0, 8.0, 9.0]}
    df_native = constructor(data)
    df_native.columns.name = "foo"  # type: ignore[union-attr]

    df = nw.from_native(df_native)

    result = df.with_columns(b=nw.col("a") + 1, c=nw.col("a") * 2).select("c", "b")

    assert result.to_native().columns.name == "foo"  # type: ignore[union-attr]
    assert result.lazy().collect(backend="pandas").to_native().columns.name == "foo"
