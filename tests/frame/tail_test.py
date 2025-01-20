from __future__ import annotations

from contextlib import nullcontext as does_not_raise

import pytest

import narwhals.stable.v1 as nw
from tests.utils import Constructor
from tests.utils import assert_equal_data


def test_tail(request: pytest.FixtureRequest, constructor: Constructor) -> None:
    if "pyspark" in str(constructor):
        request.applymarker(pytest.mark.xfail)

    data = {"a": [1, 3, 2], "b": [4, 4, 6], "z": [7.0, 8.0, 9.0]}
    expected = {"a": [3, 2], "b": [4, 6], "z": [8.0, 9]}

    df_raw = constructor(data)
    df = nw.from_native(df_raw).lazy()

    context = (
        pytest.raises(
            NotImplementedError,
            match="`LazyFrame.tail` is not supported for Dask backend with multiple partitions.",
        )
        if "dask_lazy_p2" in str(constructor)
        else does_not_raise()
    )

    with context:
        result = df.tail(2)
        assert_equal_data(result, expected)

        result = df.collect().tail(2)  # type: ignore[assignment]
        assert_equal_data(result, expected)

        result = df.collect().tail(-1)  # type: ignore[assignment]
        assert_equal_data(result, expected)

        result = df.collect().select(nw.col("a").tail(2))  # type: ignore[assignment]
        assert_equal_data(result, {"a": expected["a"]})
