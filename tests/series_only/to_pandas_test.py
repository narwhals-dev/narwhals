from __future__ import annotations

import datetime as dt
from typing import TYPE_CHECKING, Any

import pandas as pd
import pytest

import narwhals as nw
from tests.utils import PANDAS_VERSION, assert_equal_data

if TYPE_CHECKING:
    from collections.abc import Container

    from narwhals._arrow.namespace import ArrowNamespace
    from narwhals.typing import IntoDType
    from tests.utils import ConstructorEager

PANDAS_LT_1_5 = PANDAS_VERSION < (1, 5, 0)
PANDAS_LT_2 = PANDAS_VERSION < (2, 0, 0)


@pytest.mark.skipif(PANDAS_LT_2, reason="too old for pyarrow")
def test_convert(
    request: pytest.FixtureRequest, constructor_eager: ConstructorEager
) -> None:
    data = [1, 3, 2]
    if any(
        cname in str(constructor_eager)
        for cname in ("pandas_nullable", "pandas_pyarrow", "modin_pyarrow")
    ):
        request.applymarker(pytest.mark.xfail)

    series = nw.from_native(constructor_eager({"a": data}), eager_only=True)["a"].alias(
        "a"
    )

    result = series.to_pandas()
    pd.testing.assert_series_equal(result, pd.Series([1, 3, 2], name="a"))


def dtype_struct(a_dtype: IntoDType, b_dtype: IntoDType) -> nw.Struct:
    return nw.Struct({"a": a_dtype, "b": b_dtype})


@pytest.mark.skipif(PANDAS_LT_1_5, reason="too old for pyarrow")
@pytest.mark.parametrize(
    ("a", "a_dtype", "b", "b_dtype"),
    [
        ([1, 3, 8], nw.Int64, [4.1, 2.3, 3.0], nw.Float64),
        (
            [dt.datetime(2000, 1, 1), dt.datetime(2000, 1, 2)],
            nw.Datetime(),
            ["one", None],
            nw.String,
        ),
    ],
)
@pytest.mark.parametrize("nested_dtype", [nw.Struct, nw.Array])
def test_pyarrow_to_pandas_use_pyarrow(
    a: list[Any],
    a_dtype: IntoDType,
    b: list[Any],
    b_dtype: IntoDType,
    nested_dtype: type[nw.Struct | nw.Array],
    arrow_namespace: ArrowNamespace,
) -> None:
    pytest.importorskip("pyarrow")
    import pyarrow as pa

    data: dict[str, Any] = {"a": a, "b": b}
    struct_compliant = arrow_namespace.from_native(pa.table(data)).to_struct("c")
    base = dtype_struct(a_dtype, b_dtype)

    if nested_dtype is nw.Struct:
        expected_dtype: IntoDType = base
        ser_pa = struct_compliant.to_narwhals()
    else:
        expected_dtype = nw.Array(base, 1)
        list_array = pa.FixedSizeListArray.from_arrays(
            struct_compliant.native.combine_chunks(), 1
        )
        ser_pa = nw.from_native(pa.chunked_array([list_array]), series_only=True)
    ser_pd = nw.from_native(
        ser_pa.to_pandas(use_pyarrow_extension_array=True), series_only=True
    )
    assert ser_pd.dtype == expected_dtype
    assert ser_pd.dtype == ser_pa.dtype


@pytest.mark.skipif(PANDAS_LT_1_5, reason="too old for pyarrow")
@pytest.mark.parametrize(
    ("data", "pandas_dtypes"),
    [
        ([3.2, None, 1, 42.0, 99], ["double[pyarrow]"]),
        ([None, None, 10, 20], ["int64[pyarrow]"]),
        (["hello", None, "again"], ["string[pyarrow]", "large_string[pyarrow]"]),
        (
            [dt.datetime(1980, 1, 1), None],
            ["timestamp[ns][pyarrow]", "timestamp[us][pyarrow]"],
        ),
        ([dt.date(1968, 1, 1), dt.date(1992, 1, 1)], ["date32[day][pyarrow]"]),
    ],
    ids=str,
)
def test_to_pandas_use_pyarrow(
    constructor_eager: ConstructorEager,
    data: list[Any],
    pandas_dtypes: Container[str],
    request: pytest.FixtureRequest,
) -> None:
    pytest.importorskip("pyarrow")
    request.applymarker(
        pytest.mark.xfail(
            PANDAS_LT_2 and is_pandas_non_pyarrow(constructor_eager),
            reason="no `dtype_backend` arg in `convert_dtypes`",
        )
    )
    request.applymarker(
        pytest.mark.xfail(
            ("date32[day][pyarrow]" in pandas_dtypes and is_pandas(constructor_eager)),
            reason="`date` converted to `object`",
        )
    )
    name = "a"
    expected = {name: data}
    series = nw.from_native(constructor_eager(expected)).get_column(name)
    result = series.to_pandas(use_pyarrow_extension_array=True)
    actual_name = result.dtype.name
    assert actual_name in pandas_dtypes
    assert_equal_data(nw.from_native(result, series_only=True).to_frame(), expected)


def is_pandas_non_pyarrow(constructor_eager: ConstructorEager) -> bool:
    return constructor_eager.__name__ in {
        "pandas_nullable_constructor",
        "pandas_constructor",
        "modin_constructor",
    }


def is_pandas(constructor_eager: ConstructorEager) -> bool:
    return constructor_eager.__name__ in {
        "pandas_nullable_constructor",
        "pandas_constructor",
        "pandas_pyarrow_constructor",
        "modin_pyarrow_constructor",
        "modin_constructor",
    }
