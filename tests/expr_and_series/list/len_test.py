from __future__ import annotations

import pandas as pd
import pytest

import narwhals as nw
from tests.utils import PANDAS_VERSION, Constructor, ConstructorEager, assert_equal_data

data = {"a": [[1, 2], [3, 4, None], None, [], [None]]}
expected = {"a": [2, 3, None, 0, 1]}


def test_len_expr(request: pytest.FixtureRequest, constructor: Constructor) -> None:
    if any(backend in str(constructor) for backend in ("dask", "modin", "cudf")):
        request.applymarker(pytest.mark.xfail)

    if "pandas" in str(constructor) and PANDAS_VERSION < (2, 2):
        pytest.skip()

    result = nw.from_native(constructor(data)).select(
        nw.col("a").cast(nw.List(nw.Int32())).list.len()
    )

    assert_equal_data(result, expected)


def test_len_series(
    request: pytest.FixtureRequest, constructor_eager: ConstructorEager
) -> None:
    if any(backend in str(constructor_eager) for backend in ("modin", "cudf")):
        request.applymarker(pytest.mark.xfail)

    if "pandas" in str(constructor_eager) and PANDAS_VERSION < (2, 2):
        pytest.skip()

    df = nw.from_native(constructor_eager(data), eager_only=True)

    result = df["a"].cast(nw.List(nw.Int32())).list.len()
    assert_equal_data({"a": result}, expected)


def test_pandas_preserve_index(request: pytest.FixtureRequest) -> None:
    if PANDAS_VERSION < (2, 2):
        request.applymarker(pytest.mark.xfail)

    index = pd.Index(["a", "b", "c", "d", "e"])
    df = nw.from_native(pd.DataFrame(data, index=index), eager_only=True)

    result = df["a"].cast(nw.List(nw.Int32())).list.len()
    assert_equal_data({"a": result}, expected)
    assert (result.to_native().index == index).all()


def test_pandas_object_series() -> None:
    import pandas as pd

    import narwhals as nw

    s_native = pd.Series(data=data["a"])
    s = nw.from_native(s_native, series_only=True)

    with pytest.raises(TypeError):
        s.list.len()
