from __future__ import annotations

import pytest
from tests.utils import PANDAS_VERSION, Constructor, ConstructorEager, assert_equal_data

import narwhals as nw

NON_NULLABLE_CONSTRUCTOR_NAMES = {
    "pandas_constructor",
    "dask_lazy_p1_constructor",
    "dask_lazy_p2_constructor",
    "modin_constructor",
}


def test_fill_nan(request: pytest.FixtureRequest, constructor: Constructor) -> None:
    if "cudf" in str(constructor):
        request.applymarker(
            pytest.mark.xfail(
                reason="https://github.com/narwhals-dev/narwhals/issues/3231",
                raises=NotImplementedError,
            )
        )
    data_na = {"int": [-1, 1, None]}
    df = nw.from_native(constructor(data_na)).select(
        float=nw.col("int").cast(nw.Float64), float_na=nw.col("int") ** 0.5
    )
    result = df.select(nw.all().fill_nan(None))
    expected = {"float": [-1.0, 1.0, None], "float_na": [None, 1.0, None]}
    assert_equal_data(result, expected)
    assert result.lazy().collect()["float_na"].null_count() == 2
    result = df.select(nw.all().fill_nan(3.0))
    if constructor.__name__ in NON_NULLABLE_CONSTRUCTOR_NAMES:
        # no nan vs null distinction
        expected = {"float": [-1.0, 1.0, 3.0], "float_na": [3.0, 1.0, 3.0]}
        assert result.lazy().collect()["float_na"].null_count() == 0
    elif "pandas" in str(constructor) and PANDAS_VERSION >= (3,):
        expected = {"float": [-1.0, 1.0, None], "float_na": [None, 1.0, None]}
        assert result.lazy().collect()["float_na"].null_count() == 2
    else:
        expected = {"float": [-1.0, 1.0, None], "float_na": [3.0, 1.0, None]}
        assert result.lazy().collect()["float_na"].null_count() == 1
    assert_equal_data(result, expected)


def test_fill_nan_series(constructor_eager: ConstructorEager) -> None:
    data_na = {"int": [-1, 1, None]}
    s = nw.from_native(constructor_eager(data_na)).select(float_na=nw.col("int") ** 0.5)[
        "float_na"
    ]
    result = s.fill_nan(999)
    if constructor_eager.__name__ in NON_NULLABLE_CONSTRUCTOR_NAMES:
        # no nan vs null distinction
        assert_equal_data({"a": result}, {"a": [999.0, 1.0, 999.0]})
    elif "pandas" in str(constructor_eager) and PANDAS_VERSION >= (3,):
        assert_equal_data({"a": result}, {"a": [None, 1.0, None]})
    else:
        assert_equal_data({"a": result}, {"a": [999.0, 1.0, None]})
