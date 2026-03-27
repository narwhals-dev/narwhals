from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING, cast

import pytest

import narwhals as nw
from tests.utils import (
    PANDAS_VERSION,
    PYARROW_VERSION,
    Constructor,
    ConstructorEager,
    assert_equal_data,
)

if TYPE_CHECKING:
    from narwhals._native import NativeSQLFrame

data = {
    "user": [{"id": 0, "name": "john"}, {"id": 1, "name": "jane"}],
    "psw": [
        {"hash": "fake-hash-1", "ts": datetime(2026, 1, 1, 0, 0)},
        {"hash": "fake-hash-2", "ts": datetime(2026, 1, 2, 0, 0)},
    ],
}

user_dtype = nw.Struct({"id": nw.Int16(), "name": nw.String()})
psw_dtype = nw.Struct({"hash": nw.String(), "ts": nw.Datetime()})


def _spark_to_struct(native_df: NativeSQLFrame) -> NativeSQLFrame:  # pragma: no cover
    """Convert pyspark MAP<STRING, STRING> columns to proper struct columns.

    PySpark natively maps dict input to MAP<STRING, STRING>, so we need to
    reconstruct the struct columns with the correct types before casting.
    """
    _tmp_nw_compliant_frame = nw.from_native(native_df)._compliant_frame
    F = _tmp_nw_compliant_frame._F  # type: ignore[attr-defined]
    T = _tmp_nw_compliant_frame._native_dtypes  # type: ignore[attr-defined]  # noqa: N806

    return native_df.withColumns(
        {
            "user": F.struct(
                F.col("user.id").cast(T.IntegerType()).alias("id"),
                F.col("user.name").cast(T.StringType()).alias("name"),
            ),
            "psw": F.struct(
                F.col("psw.hash").cast(T.StringType()).alias("hash"),
                F.col("psw.ts").cast(T.TimestampType()).alias("ts"),
            ),
        }
    )


def test_unnest_expr(request: pytest.FixtureRequest, constructor: Constructor) -> None:
    if any(backend in str(constructor) for backend in ("dask", "sqlframe")):
        request.applymarker(pytest.mark.xfail)

    if "pandas" in str(constructor) and (
        PANDAS_VERSION < (2, 2, 0) or PYARROW_VERSION == (0, 0, 0)
    ):
        pytest.skip()

    native_df = constructor(data)
    if "spark" in str(constructor):  # pragma: no cover
        native_df = _spark_to_struct(cast("NativeSQLFrame", native_df))

    df = nw.from_native(native_df).select(user=nw.col("user").cast(user_dtype))

    result = df.select(nw.col("user").struct.unnest())
    expected = {"id": [0, 1], "name": ["john", "jane"]}
    assert_equal_data(result, expected)


def test_unnest_expr_multi(
    request: pytest.FixtureRequest, constructor: Constructor
) -> None:
    if any(backend in str(constructor) for backend in ("dask", "sqlframe")):
        request.applymarker(pytest.mark.xfail)

    if "pandas" in str(constructor) and (
        PANDAS_VERSION < (2, 2, 0) or PYARROW_VERSION == (0, 0, 0)
    ):
        pytest.skip()

    native_df = constructor(data)
    if "spark" in str(constructor):  # pragma: no cover
        native_df = _spark_to_struct(cast("NativeSQLFrame", native_df))

    df = nw.from_native(native_df).select(
        user=nw.col("user").cast(user_dtype), psw=nw.col("psw").cast(psw_dtype)
    )

    result = df.select(nw.col("user", "psw").struct.unnest())
    expected = {
        "id": [0, 1],
        "name": ["john", "jane"],
        "hash": ["fake-hash-1", "fake-hash-2"],
        "ts": [datetime(2026, 1, 1, 0, 0), datetime(2026, 1, 2, 0, 0)],
    }
    assert_equal_data(result, expected)


def test_unnest_series(constructor_eager: ConstructorEager) -> None:
    if "pandas" in str(constructor_eager) and (
        PANDAS_VERSION < (2, 2, 0) or PYARROW_VERSION == (0, 0, 0)
    ):
        pytest.skip()

    df = nw.from_native(constructor_eager(data), eager_only=True).select(
        user=nw.col("user").cast(user_dtype)
    )

    result = df.get_column("user").struct.unnest()
    expected = {"id": [0, 1], "name": ["john", "jane"]}
    assert_equal_data(result, expected)
