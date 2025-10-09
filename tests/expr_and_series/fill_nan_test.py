from __future__ import annotations

import narwhals as nw
from tests.conftest import (
    dask_lazy_p1_constructor,
    dask_lazy_p2_constructor,
    modin_constructor,
    pandas_constructor,
)
from tests.utils import PANDAS_VERSION, Constructor, ConstructorEager, assert_equal_data

NON_NULLABLE_CONSTRUCTORS = [
    pandas_constructor,
    dask_lazy_p1_constructor,
    dask_lazy_p2_constructor,
    modin_constructor,
]


def test_fill_nan(constructor: Constructor) -> None:
    data_na = {"int": [-1, 1, None]}
    df = nw.from_native(constructor(data_na)).select(
        float=nw.col("int").cast(nw.Float64), float_na=nw.col("int") ** 0.5
    )
    result = df.select(nw.all().fill_nan(None))
    expected = {"float": [-1.0, 1.0, None], "float_na": [None, 1.0, None]}
    assert_equal_data(result, expected)
    assert result.lazy().collect()["float_na"].null_count() == 2
    result = df.select(nw.all().fill_nan(3.0))
    if any(constructor is c for c in NON_NULLABLE_CONSTRUCTORS):
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
    if any(constructor_eager is c for c in NON_NULLABLE_CONSTRUCTORS):
        # no nan vs null distinction
        assert_equal_data({"a": result}, {"a": [999.0, 1.0, 999.0]})
    elif "pandas" in str(constructor_eager) and PANDAS_VERSION >= (3,):
        assert_equal_data({"a": result}, {"a": [None, 1.0, None]})
    else:
        assert_equal_data({"a": result}, {"a": [999.0, 1.0, None]})
