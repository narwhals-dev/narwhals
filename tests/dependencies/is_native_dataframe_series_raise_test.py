from __future__ import annotations

import re
from typing import TYPE_CHECKING
from typing import Any
from typing import Callable

import pytest

import narwhals.stable.v1 as nw
from narwhals.stable.v1.dependencies import is_cudf_dataframe
from narwhals.stable.v1.dependencies import is_cudf_series
from narwhals.stable.v1.dependencies import is_ibis_table
from narwhals.stable.v1.dependencies import is_modin_dataframe
from narwhals.stable.v1.dependencies import is_modin_series
from narwhals.stable.v1.dependencies import is_pandas_dataframe
from narwhals.stable.v1.dependencies import is_pandas_like_dataframe
from narwhals.stable.v1.dependencies import is_pandas_like_series
from narwhals.stable.v1.dependencies import is_pandas_series
from narwhals.stable.v1.dependencies import is_polars_dataframe
from narwhals.stable.v1.dependencies import is_polars_lazyframe
from narwhals.stable.v1.dependencies import is_polars_series
from narwhals.stable.v1.dependencies import is_pyarrow_chunked_array
from narwhals.stable.v1.dependencies import is_pyarrow_table

if TYPE_CHECKING:
    from tests.utils import Constructor
    from tests.utils import ConstructorEager


@pytest.mark.parametrize(
    "is_native_dataframe",
    [
        is_pandas_dataframe,
        is_modin_dataframe,
        is_polars_dataframe,
        is_cudf_dataframe,
        is_ibis_table,
        is_polars_lazyframe,
        is_pyarrow_table,
        is_pandas_like_dataframe,
    ],
)
def test_is_native_dataframe(
    constructor: Constructor, is_native_dataframe: Callable[[Any], Any]
) -> None:
    data = {"a": [1, 2], "b": ["bar", "foo"]}
    df = nw.from_native(constructor(data))
    func_name = is_native_dataframe.__name__
    msg = re.escape(
        f"You passed a `{type(df)}` to `{func_name}`.\n\n"
        f"Hint: Instead of e.g. `{func_name}(df)`, "
        f"did you mean `{func_name}(df.to_native())`?"
    )
    with pytest.raises(TypeError, match=msg):
        is_native_dataframe(df)


@pytest.mark.parametrize(
    "is_native_series",
    [
        is_pandas_series,
        is_modin_series,
        is_polars_series,
        is_cudf_series,
        is_pyarrow_chunked_array,
        is_pandas_like_series,
    ],
)
def test_is_native_series(
    constructor_eager: ConstructorEager, is_native_series: Callable[[Any], Any]
) -> None:
    data = {"a": [1, 2]}
    ser = nw.from_native(constructor_eager(data))["a"]
    func_name = is_native_series.__name__
    msg = re.escape(
        f"You passed a `{type(ser)}` to `{func_name}`.\n\n"
        f"Hint: Instead of e.g. `{func_name}(ser)`, "
        f"did you mean `{func_name}(ser.to_native())`?"
    )
    with pytest.raises(TypeError, match=msg):
        is_native_series(ser)
