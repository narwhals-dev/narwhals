from __future__ import annotations

import datetime as dt
from typing import TYPE_CHECKING, Any, cast

import pandas as pd
import pytest

import narwhals as nw
from narwhals._utils import Version
from tests.utils import PANDAS_VERSION

if TYPE_CHECKING:
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


@pytest.fixture(scope="session")
def arrow_namespace() -> ArrowNamespace:
    """Using for `ArrowDataFrame.to_struct`.

    Has a backcompat path, but ideally we replace this if/when [`nw.DataFrame.to_struct`] is added.

    [`nw.DataFrame.to_struct`]: https://github.com/narwhals-dev/narwhals/pull/2839#issuecomment-3110332853
    """
    pytest.importorskip("pyarrow")
    return Version.MAIN.namespace.from_backend("pyarrow").compliant


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
