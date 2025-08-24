from __future__ import annotations

import narwhals as nw
from tests.utils import Constructor, ConstructorEager, assert_equal_data


def test_is_nan(constructor: Constructor) -> None:
    df = nw.from_native(
        constructor({"a": [1.1, 2.0, float("nan")], "b": [3.1, 4.0, None]})
    )

    if df.implementation.is_dask():
        # test both pyarrow-dtypes and numpy-dtypes
        df = nw.from_native(
            df.to_native().astype({"a": "Float64[pyarrow]", "b": "float64"})  # type: ignore[attr-defined]
        )

    result = df.select(nw.all().fill_nan(None))
    expected = {"a": [1.1, 2.0, None], "b": [3.1, 4.0, None]}
    assert_equal_data(result, expected)
    assert result.lazy().collect()["a"].null_count() == 1
    result = df.select(nw.all().fill_nan(3.0))
    if any(x in str(constructor) for x in ("pandas", "dask", "cudf", "modin")):
        # pandas doesn't distinguish nan vs null
        expected = {"a": [1.1, 2.0, 3.0], "b": [3.1, 4.0, 3.0]}
        assert int(result.lazy().collect()["b"].null_count()) == 0
    else:
        expected = {"a": [1.1, 2.0, 3.0], "b": [3.1, 4.0, None]}
        assert result.lazy().collect()["b"].null_count() == 1
    assert_equal_data(result, expected)
    assert result.lazy().collect()["a"].null_count() == 0


def test_is_nan_series(constructor_eager: ConstructorEager) -> None:
    s = nw.from_native(
        constructor_eager({"a": [1.1, 2.0, float("nan")]}), eager_only=True
    )["a"]

    result = s.fill_nan(999)
    assert_equal_data({"a": result}, {"a": [1.1, 2.0, 999]})
