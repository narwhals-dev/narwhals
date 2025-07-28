from __future__ import annotations

import datetime as dt
from typing import TYPE_CHECKING, Any

import pandas as pd
import pytest
from pandas.testing import assert_series_equal

import narwhals as nw
from tests.utils import PANDAS_VERSION

if TYPE_CHECKING:
    from narwhals._arrow.namespace import ArrowNamespace
    from narwhals.typing import IntoDType
    from tests.utils import ConstructorEager


@pytest.mark.skipif(PANDAS_VERSION < (2, 0, 0), reason="too old for pyarrow")
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
    assert_series_equal(result, pd.Series([1, 3, 2], name="a"))


def dtype_struct(a_dtype: IntoDType, b_dtype: IntoDType) -> nw.Struct:
    return nw.Struct({"a": a_dtype, "b": b_dtype})


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
