from __future__ import annotations

import datetime as dt
from itertools import chain
from typing import TYPE_CHECKING, Any, cast

import pandas as pd
import pytest

import narwhals as nw
from narwhals.exceptions import InvalidOperationError
from narwhals.typing import ToPandasArrowKwds
from tests.utils import (
    PANDAS_LT_1_5,
    PANDAS_LT_2,
    PANDAS_VERSION,
    assert_equal_data,
    is_pandas,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

    from narwhals._arrow.namespace import ArrowNamespace
    from narwhals.typing import IntoDType
    from tests.utils import ConstructorEager


@pytest.mark.filterwarnings("ignore:.*Passing a BlockManager.*:DeprecationWarning")
@pytest.mark.skipif(PANDAS_VERSION < (2, 0, 0), reason="too old for pandas-pyarrow")
def test_convert_pandas(constructor_eager: ConstructorEager) -> None:
    data = {"a": [1, 3, 2], "b": [4, 4, 6], "z": [7.0, 8.0, 9.0]}
    df_raw = constructor_eager(data)
    result = nw.from_native(df_raw, eager_only=True).to_pandas()

    if constructor_eager.__name__.startswith("pandas"):
        expected = cast("pd.DataFrame", constructor_eager(data))
    elif "modin_pyarrow" in str(constructor_eager):
        expected = pd.DataFrame(data).convert_dtypes(dtype_backend="pyarrow")
    else:
        expected = pd.DataFrame(data)

    pd.testing.assert_frame_equal(result, expected)


def schema_struct(a_dtype: IntoDType, b_dtype: IntoDType) -> nw.Schema:
    return nw.Schema({"c": nw.Struct({"a": a_dtype, "b": b_dtype})})


def schema_list(a_dtype: IntoDType, b_dtype: IntoDType) -> nw.Schema:
    return nw.Schema({"a": nw.List(a_dtype), "b": nw.List(b_dtype)})


@pytest.mark.skipif(PANDAS_VERSION < (1, 5, 0), reason="too old for pyarrow")
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
@pytest.mark.parametrize("nested_dtype", [nw.Struct, nw.List])
def test_pyarrow_to_pandas_use_pyarrow(
    a: list[Any],
    a_dtype: IntoDType,
    b: list[Any],
    b_dtype: IntoDType,
    nested_dtype: type[nw.Struct | nw.List],
    arrow_namespace: ArrowNamespace,
) -> None:
    pytest.importorskip("pyarrow")
    import pyarrow as pa

    if nested_dtype is nw.Struct:
        expected_schema = schema_struct(a_dtype, b_dtype)
        data: dict[str, Any] = {"a": a, "b": b}
        df_pa = (
            arrow_namespace.from_native(pa.table(data))
            .to_struct("c")
            .to_frame()
            .to_narwhals()
        )
    else:
        expected_schema = schema_list(a_dtype, b_dtype)
        data = {"a": [pa.scalar(a)], "b": [pa.scalar(b)]}
        df_pa = nw.from_native(pa.table(data))
    df_pd = nw.from_native(df_pa.to_pandas(use_pyarrow_extension_array=True))
    assert df_pd.schema == expected_schema
    assert df_pd.schema == df_pa.schema


@pytest.mark.skipif(PANDAS_LT_1_5, reason="too old for pyarrow")
@pytest.mark.parametrize("kwds", [{}, ToPandasArrowKwds(self_destruct=True)])
@pytest.mark.parametrize(
    ("data", "pandas_dtypes"),
    [
        (
            {"a": [3.2, None, 42.0, 99], "b": [None, None, 10, 20]},
            {"a": ["double[pyarrow]"], "b": ["int64[pyarrow]"]},
        ),
        (
            {"a": ["hello", None, "again"], "b": ["a", "b", "c"]},
            {
                "a": ["string[pyarrow]", "large_string[pyarrow]"],
                "b": ["string[pyarrow]", "large_string[pyarrow]"],
            },
        ),
        (
            {
                "a": [dt.datetime(1980, 1, 1), None],
                "b": [dt.datetime(2001, 1, 1), dt.datetime(2002, 1, 1, 3, 1, 2)],
            },
            {
                "a": ["timestamp[ns][pyarrow]", "timestamp[us][pyarrow]"],
                "b": ["timestamp[ns][pyarrow]", "timestamp[us][pyarrow]"],
            },
        ),
        (
            {
                "a": [None, dt.date(1982, 1, 1)],
                "b": [dt.date(1968, 1, 1), dt.date(1992, 1, 1)],
            },
            {"a": ["date32[day][pyarrow]"], "b": ["date32[day][pyarrow]"]},
        ),
        (
            {"a": [None, True], "b": [True, False]},
            {"a": ["bool[pyarrow]"], "b": ["bool[pyarrow]"]},
        ),
        ({"a": [b"hi"]}, {"a": ["binary[pyarrow]", "large_binary[pyarrow]"]}),
    ],
)
def test_to_pandas_use_pyarrow(
    constructor_eager: ConstructorEager,
    data: dict[str, list[Any]],
    pandas_dtypes: dict[str, Sequence[str]],
    kwds: ToPandasArrowKwds,
    request: pytest.FixtureRequest,
) -> None:
    pytest.importorskip("pyarrow")
    request.applymarker(
        pytest.mark.xfail(
            is_pandas(constructor_eager) and bool(kwds),
            reason="Only `convert_dtypes` behavior is supported for pandas-like",
            raises=InvalidOperationError,
        )
    )
    request.applymarker(
        pytest.mark.xfail(
            PANDAS_LT_2
            and is_pandas(
                constructor_eager,
                exclude={
                    "pandas_pyarrow_constructor",
                    "modin_pyarrow_constructor",
                    "cudf_constructor",
                },
            ),
            reason="no `dtype_backend` arg in `convert_dtypes`",
            raises=TypeError,
        )
    )
    pandas_unsupported = {
        "date32[day][pyarrow]",
        "binary[pyarrow]",
        "large_binary[pyarrow]",
    }

    request.applymarker(
        pytest.mark.xfail(
            bool(
                is_pandas(constructor_eager)
                and pandas_unsupported.intersection(
                    chain.from_iterable(pandas_dtypes.values())
                )
            ),
            reason="`date` converted to `object`",
            raises=AssertionError,
        )
    )
    frame = nw.from_native(constructor_eager(data))
    result = frame.to_pandas(use_pyarrow_extension_array=True, **kwds)
    for column, dtypes in pandas_dtypes.items():
        actual_name = result[column].dtype.name
        assert actual_name in dtypes
    assert_equal_data(result, data)


@pytest.mark.skipif(not PANDAS_LT_1_5)
def test_to_pandas_no_arrow_dtype() -> None:  # pragma: no cover
    pytest.importorskip("pyarrow")
    import pyarrow as pa

    data: dict[str, Any] = {"a": [1, 2, 3], "b": ["four", "five", "six"]}
    df_pa = nw.from_native(pa.table(data))
    with pytest.raises(NotImplementedError, match="1.5"):
        df_pa.to_pandas(use_pyarrow_extension_array=True)
