from __future__ import annotations

import pytest

import narwhals as nw
from tests.utils import PANDAS_VERSION, Constructor, ConstructorEager, assert_equal_data

data = {"a": [[1, 2], [3, 4, None], None, [], [None]]}
expected = {"a": [2, 3, None, 0, 1]}


def test_len_expr(
    request: pytest.FixtureRequest, nw_frame_constructor: Constructor
) -> None:
    if any(backend in str(nw_frame_constructor) for backend in ("dask", "cudf")):
        request.applymarker(pytest.mark.xfail)

    if "pandas" in str(nw_frame_constructor):
        if PANDAS_VERSION < (2, 2):
            pytest.skip()
        pytest.importorskip("pyarrow")

    result = nw.from_native(nw_frame_constructor(data)).select(
        nw.col("a").cast(nw.List(nw.Int32())).list.len()
    )

    assert_equal_data(result, expected)


def test_len_series(
    request: pytest.FixtureRequest, nw_eager_constructor: ConstructorEager
) -> None:
    if "cudf" in str(nw_eager_constructor):
        request.applymarker(pytest.mark.xfail)

    if "pandas" in str(nw_eager_constructor):
        if PANDAS_VERSION < (2, 2):
            pytest.skip()
        pytest.importorskip("pyarrow")

    df = nw.from_native(nw_eager_constructor(data), eager_only=True)

    result = df["a"].cast(nw.List(nw.Int32())).list.len()
    assert_equal_data({"a": result}, expected)


def test_pandas_preserve_index(request: pytest.FixtureRequest) -> None:
    pytest.importorskip("pandas")
    pytest.importorskip("pyarrow")
    import pandas as pd

    if PANDAS_VERSION < (2, 2):
        request.applymarker(pytest.mark.xfail)

    index = pd.Index(["a", "b", "c", "d", "e"])
    df = nw.from_native(pd.DataFrame(data, index=index), eager_only=True)

    result = df["a"].cast(nw.List(nw.Int32())).list.len()
    assert_equal_data({"a": result}, expected)
    assert (result.to_native().index == index).all()


def test_pandas_object_series() -> None:
    pytest.importorskip("pandas")
    import pandas as pd

    s_native = pd.Series(data=data["a"])
    s = nw.from_native(s_native, series_only=True)

    with pytest.raises(TypeError):
        s.list.len()
