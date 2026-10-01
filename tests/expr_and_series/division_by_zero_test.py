from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pytest

import narwhals as nw
from tests.utils import (
    PANDAS_VERSION,
    POLARS_VERSION,
    PYARROW_VERSION,
    Constructor,
    ConstructorEager,
    assert_equal_data,
)

if TYPE_CHECKING:
    from collections.abc import Callable

    from narwhals.typing import DTypeBackend

data: dict[str, list[float]] = {
    "int": [-2, 0, 2],
    "float": [-2.1, 0.0, 2.1],
    "denominator": [0, 0, 0],
}
expected_truediv: list[float] = [float("-inf"), float("nan"), float("inf")]
expected_floordiv: list[float | None] = [None, None, None]


@pytest.mark.parametrize("get_denominator", [lambda _: 0, lambda df: df["denominator"]])
def test_series_truediv_by_zero(
    constructor_eager: ConstructorEager,
    get_denominator: Callable[[nw.DataFrame[Any]], int | nw.Series[Any]],
) -> None:
    df = nw.from_native(constructor_eager(data), eager_only=True)
    denominator = get_denominator(df)
    result = {"int": df["int"] / denominator, "float": df["float"] / denominator}
    expected = {"int": expected_truediv, "float": expected_truediv}
    assert_equal_data(result, expected)


@pytest.mark.parametrize("denominator", [0, nw.lit(0), nw.col("denominator")])
def test_expr_truediv_by_zero(
    constructor: Constructor, denominator: int | nw.Expr
) -> None:
    df = nw.from_native(constructor(data))
    result = df.select(nw.col("int", "float") / denominator)
    expected = {"int": expected_truediv, "float": expected_truediv}
    assert_equal_data(result, expected)


@pytest.mark.parametrize("get_denominator", [lambda _: 0, lambda df: df["denominator"]])
def test_series_floordiv_by_zero(
    constructor_eager: ConstructorEager,
    request: pytest.FixtureRequest,
    get_denominator: Callable[[nw.DataFrame[Any]], int | nw.Series[Any]],
) -> None:
    df = nw.from_native(constructor_eager(data), eager_only=True)

    if "polars" in str(constructor_eager) and POLARS_VERSION < (0, 20, 7):
        pytest.skip(reason="bug")
    if "cudf" in str(constructor_eager):
        request.applymarker(pytest.mark.xfail)

    denominator = get_denominator(df)
    result = {"result": df["int"] // denominator}
    expected = {"result": expected_floordiv}
    assert_equal_data(result, expected)


@pytest.mark.parametrize("denominator", [0, nw.lit(0), nw.col("denominator")])
def test_expr_floordiv_by_zero(
    constructor: Constructor, request: pytest.FixtureRequest, denominator: int | nw.Expr
) -> None:
    df = nw.from_native(constructor(data))

    if "polars" in str(constructor) and POLARS_VERSION < (0, 20, 7):
        pytest.skip(reason="bug")
    if "cudf" in str(constructor):
        request.applymarker(pytest.mark.xfail)

    result = df.select(result=nw.col("int") // denominator)
    expected = {"result": expected_floordiv}
    assert_equal_data(result, expected)
    assert_equal_data(result.select(nw.col("result").is_null().all()), {"result": [True]})


@pytest.mark.parametrize(
    ("numerator", "expected"),
    list(
        zip(
            [*data["int"], *data["float"]],
            [*expected_truediv, *expected_truediv],
            strict=True,
        )
    ),
)
def test_series_rtruediv_by_zero(
    constructor_eager: ConstructorEager, numerator: float, expected: float
) -> None:
    df = nw.from_native(constructor_eager(data), eager_only=True)
    result = {"result": numerator / df["denominator"]}
    assert_equal_data(result, {"result": [expected] * len(df)})


@pytest.mark.parametrize(
    ("numerator", "expected"),
    list(
        zip(
            [*data["int"], *data["float"]],
            [*expected_truediv, *expected_truediv],
            strict=True,
        )
    ),
)
def test_expr_rtruediv_by_zero(
    constructor: Constructor, numerator: float, expected: float
) -> None:
    df = nw.from_native(constructor(data))
    result = df.select(result=numerator / nw.col("denominator"))
    assert_equal_data(result, {"result": [expected] * len(data["denominator"])})
    assert_equal_data(result.select((~nw.all().is_finite()).all()), {"result": [True]})


@pytest.mark.parametrize("numerator", data["int"])
def test_series_rfloordiv_by_zero(
    constructor_eager: ConstructorEager, request: pytest.FixtureRequest, numerator: float
) -> None:
    if "polars" in str(constructor_eager) and POLARS_VERSION < (0, 20, 7):
        pytest.skip(reason="bug")
    if "cudf" in str(constructor_eager):
        request.applymarker(pytest.mark.xfail)

    df = nw.from_native(constructor_eager(data), eager_only=True)

    result = {"result": numerator // df["denominator"]}
    assert_equal_data(result, {"result": expected_floordiv})


@pytest.mark.parametrize("numerator", data["int"])
def test_expr_rfloordiv_by_zero(
    constructor: Constructor, request: pytest.FixtureRequest, numerator: float
) -> None:
    if "polars" in str(constructor) and POLARS_VERSION < (0, 20, 7):
        pytest.skip(reason="bug")
    if "cudf" in str(constructor):
        request.applymarker(pytest.mark.xfail)

    df = nw.from_native(constructor(data))

    result = df.select(result=numerator // nw.col("denominator"))
    expected = {"result": expected_floordiv}
    assert_equal_data(result, expected)


# Floor division by zero on pandas-like backends must keep the dtype backend of the
# input: nullable stays nullable, pyarrow-backed stays pyarrow-backed. numpy-backed
# integers cannot hold missing values, so they become float64 with NaN.
PYARROW_UNAVAILABLE = PYARROW_VERSION == (0, 0, 0)
require_pd_2_1 = pytest.mark.skipif(
    PANDAS_VERSION < (2, 1, 0) or PYARROW_UNAVAILABLE,
    reason="pyarrow-backed dtypes require pandas>=2.1 and pyarrow",
)
require_pd_2_0 = pytest.mark.skipif(
    PANDAS_VERSION < (2, 0, 0), reason="nullable dtypes require pandas>=2.0"
)
DTYPE_BACKENDS = [
    pytest.param("pyarrow", marks=require_pd_2_1),
    pytest.param("numpy_nullable", marks=require_pd_2_0),
    None,
]
# (input dtype, expected output dtype) per dtype backend and numeric kind.
FLOORDIV_DTYPES: dict[DTypeBackend, dict[str, tuple[str, str]]] = {
    None: {"int": ("int64", "float64"), "float": ("float64", "float64")},
    "numpy_nullable": {"int": ("Int64", "Int64"), "float": ("Float64", "Float64")},
    "pyarrow": {
        "int": ("int64[pyarrow]", "int64[pyarrow]"),
        "float": ("double[pyarrow]", "double[pyarrow]"),
    },
}
INDEX = [8, 7, 6]


@pytest.mark.parametrize("dtype_backend", DTYPE_BACKENDS)
@pytest.mark.parametrize("kind", ["int", "float"])
def test_floordiv_by_zero_keeps_dtype_backend_pandas(
    dtype_backend: DTypeBackend, kind: str
) -> None:
    pytest.importorskip("pandas")
    import pandas as pd

    in_dtype, out_dtype = FLOORDIV_DTYPES[dtype_backend][kind]
    numerator = pd.Series([-3, 0, 5], name="a", index=INDEX).astype(in_dtype)
    denominator = pd.Series([0, 2, 2], name="b", index=INDEX).astype(in_dtype)
    series = nw.from_native(numerator, series_only=True)

    result = (series // nw.from_native(denominator, series_only=True)).to_native()
    expected = pd.Series([None, 0, 2], name="a", index=INDEX, dtype=out_dtype)
    pd.testing.assert_series_equal(result, expected)

    result = (series // 0).to_native()
    expected = pd.Series([None] * 3, name="a", index=INDEX, dtype=out_dtype)
    pd.testing.assert_series_equal(result, expected)


@pytest.mark.parametrize("dtype_backend", DTYPE_BACKENDS)
@pytest.mark.parametrize("kind", ["int", "float"])
def test_rfloordiv_by_zero_keeps_dtype_backend_pandas(
    dtype_backend: DTypeBackend, kind: str
) -> None:
    pytest.importorskip("pandas")
    import pandas as pd

    in_dtype, out_dtype = FLOORDIV_DTYPES[dtype_backend][kind]
    denominator = pd.Series([0, 2, -3], name="a", index=INDEX).astype(in_dtype)
    series = nw.from_native(denominator, series_only=True)

    result = (7 // series).to_native()
    expected = pd.Series([None, 3, -3], name="a", index=INDEX, dtype=out_dtype)
    pd.testing.assert_series_equal(result, expected)


@pytest.mark.parametrize("dtype_backend", DTYPE_BACKENDS)
def test_expr_floordiv_by_zero_keeps_dtype_backend_pandas(
    dtype_backend: DTypeBackend,
) -> None:
    pytest.importorskip("pandas")
    import pandas as pd

    dtypes = FLOORDIV_DTYPES[dtype_backend]
    native = pd.DataFrame(
        {"int": [-3, 0, 5], "float": [-3.5, 0.0, 5.5], "denominator": [0, 2, 2]},
        index=INDEX,
    ).astype(
        {
            "int": dtypes["int"][0],
            "float": dtypes["float"][0],
            "denominator": dtypes["int"][0],
        }
    )
    df = nw.from_native(native, eager_only=True)

    result = df.select(
        int_col=nw.col("int") // nw.col("denominator"),
        int_lit=nw.col("int") // nw.lit(0),
        float_col=nw.col("float") // nw.col("denominator"),
        rfloordiv=7 // nw.col("denominator"),
    ).to_native()
    expected = pd.DataFrame(
        {
            "int_col": pd.Series([None, 0, 2], dtype=dtypes["int"][1]),
            "int_lit": pd.Series([None] * 3, dtype=dtypes["int"][1]),
            "float_col": pd.Series([None, 0.0, 2.0], dtype=dtypes["float"][1]),
            "rfloordiv": pd.Series([None, 3, 3], dtype=dtypes["int"][1]),
        }
    ).set_axis(INDEX)
    pd.testing.assert_frame_equal(result, expected)
