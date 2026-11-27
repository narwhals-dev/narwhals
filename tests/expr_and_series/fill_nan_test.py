from __future__ import annotations

import pytest
from packaging.version import Version as PkgVersion

import narwhals as nw
from tests.conftest import NON_NULLABLE_CONSTRUCTORS
from tests.utils import (
    DASK_PYARROW_DTYPES,
    PANDAS_VERSION,
    Constructor,
    ConstructorEager,
    assert_equal_data,
    xfail_if_builtin_dask,
)


@pytest.mark.parametrize("column", ["float", "float_na"])
def test_fill_nan(
    request: pytest.FixtureRequest, constructor: Constructor, column: str
) -> None:
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
    if column == "float":
        xfail_if_builtin_dask(df, request, when=DASK_PYARROW_DTYPES)
    result = df.select(nw.all().fill_nan(None)).select(column)
    expected: dict[str, list[float | None]] = {
        "float": [-1.0, 1.0, None],
        "float_na": [None, 1.0, None],
    }
    assert_equal_data(result, {column: expected[column]})
    assert result.lazy().collect()[column].null_count() == expected[column].count(None)
    result = df.select(nw.all().fill_nan(3.0)).select(column)
    if constructor in NON_NULLABLE_CONSTRUCTORS:
        # no nan vs null distinction
        expected = {"float": [-1.0, 1.0, 3.0], "float_na": [3.0, 1.0, 3.0]}
    elif PANDAS_VERSION >= PkgVersion("3") and any(
        x in str(constructor) for x in ("pandas", "dask")
    ):
        expected = {"float": [-1.0, 1.0, None], "float_na": [None, 1.0, None]}
    else:
        expected = {"float": [-1.0, 1.0, None], "float_na": [3.0, 1.0, None]}
    assert_equal_data(result, {column: expected[column]})
    assert result.lazy().collect()[column].null_count() == expected[column].count(None)


def test_fill_nan_series(constructor_eager: ConstructorEager) -> None:
    data_na = {"int": [-1, 1, None]}
    s = nw.from_native(constructor_eager(data_na)).select(float_na=nw.col("int") ** 0.5)[
        "float_na"
    ]
    result = s.fill_nan(999)
    if constructor_eager in NON_NULLABLE_CONSTRUCTORS:
        # no nan vs null distinction
        assert_equal_data({"a": result}, {"a": [999.0, 1.0, 999.0]})
    elif "pandas" in str(constructor_eager) and PANDAS_VERSION >= PkgVersion("3"):
        assert_equal_data({"a": result}, {"a": [None, 1.0, None]})
    else:
        assert_equal_data({"a": result}, {"a": [999.0, 1.0, None]})
