from __future__ import annotations

from contextlib import nullcontext as does_not_raise
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Literal

import pyarrow as pa
import pytest

import narwhals as nw
from narwhals._arrow.utils import parse_datetime_format
from narwhals._pandas_like.utils import get_dtype_backend
from tests.utils import (
    PANDAS_VERSION,
    assert_equal_data,
    is_pyarrow_windows_no_tzdata,
    is_windows,
)

if TYPE_CHECKING:
    from tests.utils import Constructor, ConstructorEager

data = {"a": ["2020-01-01T12:34:56"]}


def test_to_datetime(constructor: Constructor) -> None:
    if "cudf" in str(constructor):
        expected = "2020-01-01T12:34:56.000000000"
    else:
        expected = "2020-01-01 12:34:56"

    result = (
        nw.from_native(constructor(data))
        .lazy()
        .select(b=nw.col("a").str.to_datetime(format="%Y-%m-%dT%H:%M:%S"))
    )
    result_schema = result.collect_schema()
    assert isinstance(result_schema["b"], nw.Datetime)
    assert result_schema["b"].time_zone is None  # pyright: ignore[reportAttributeAccessIssue]
    result_item = result.collect().item(row=0, column="b")
    assert str(result_item) == expected


def test_to_datetime_series(constructor_eager: ConstructorEager) -> None:
    if "cudf" in str(constructor_eager):
        expected = "2020-01-01T12:34:56.000000000"
    else:
        expected = "2020-01-01 12:34:56"

    result = (
        nw.from_native(constructor_eager(data), eager_only=True)["a"].str.to_datetime(
            format="%Y-%m-%dT%H:%M:%S"
        )
    ).item(0)
    assert str(result) == expected


@pytest.mark.parametrize(
    ("data", "expected", "expected_cudf", "expected_pyspark"),
    [
        (
            {"a": ["2020-01-01T12:34:56"]},
            "2020-01-01 12:34:56",
            "2020-01-01T12:34:56.000000000",
            "2020-01-01 12:34:56+00:00",
        ),
        (
            {"a": ["2020-01-01T12:34"]},
            "2020-01-01 12:34:00",
            "2020-01-01T12:34:00.000000000",
            "2020-01-01 12:34:00+00:00",
        ),
        (
            {"a": ["20240101123456"]},
            "2024-01-01 12:34:56",
            "2024-01-01T12:34:56.000000000",
            "2024-01-01 12:34:56+00:00",
        ),
    ],
)
def test_to_datetime_infer_fmt(
    request: pytest.FixtureRequest,
    constructor: Constructor,
    data: dict[str, list[str]],
    expected: str,
    expected_cudf: str,
    expected_pyspark: str,
) -> None:
    if (
        ("polars" in str(constructor) and str(data["a"][0]).isdigit())
        or "duckdb" in str(constructor)
        or ("pyspark" in str(constructor) and data["a"][0] == "20240101123456")
        or "ibis" in str(constructor)
    ):
        request.applymarker(pytest.mark.xfail)

    if "cudf" in str(constructor):
        expected = expected_cudf
    elif "pyspark" in str(constructor):
        expected = expected_pyspark

    result = (
        nw.from_native(constructor(data))
        .lazy()
        .select(b=nw.col("a").str.to_datetime())
        .collect()
        .item(row=0, column="b")
    )
    assert str(result) == expected


@pytest.mark.parametrize(
    ("data", "expected", "expected_cudf"),
    [
        (
            {"a": ["2020-01-01T12:34:56"]},
            "2020-01-01 12:34:56",
            "2020-01-01T12:34:56.000000000",
        ),
        (
            {"a": ["2020-01-01T12:34"]},
            "2020-01-01 12:34:00",
            "2020-01-01T12:34:00.000000000",
        ),
        (
            {"a": ["20240101123456"]},
            "2024-01-01 12:34:56",
            "2024-01-01T12:34:56.000000000",
        ),
    ],
)
def test_to_datetime_series_infer_fmt(
    request: pytest.FixtureRequest,
    constructor_eager: ConstructorEager,
    data: dict[str, list[str]],
    expected: str,
    expected_cudf: str,
) -> None:
    if "polars" in str(constructor_eager) and str(data["a"][0]).isdigit():
        request.applymarker(pytest.mark.xfail)
    if "cudf" in str(constructor_eager):
        expected = expected_cudf

    result = (
        nw.from_native(constructor_eager(data), eager_only=True)["a"].str.to_datetime()
    ).item(0)
    assert str(result) == expected


def test_to_datetime_infer_fmt_from_date(
    constructor: Constructor, request: pytest.FixtureRequest
) -> None:
    if any(x in str(constructor) for x in ("duckdb", "ibis")):
        request.applymarker(pytest.mark.xfail)
    data = {"z": ["2020-01-01", "2020-01-02", None]}
    if "pyspark" in str(constructor):
        expected = [
            datetime(2020, 1, 1, tzinfo=timezone.utc),
            datetime(2020, 1, 2, tzinfo=timezone.utc),
            None,
        ]
    else:
        expected = [datetime(2020, 1, 1), datetime(2020, 1, 2), None]
    result = (
        nw.from_native(constructor(data)).lazy().select(nw.col("z").str.to_datetime())
    )
    assert_equal_data(result, {"z": expected})


def test_pyarrow_infer_datetime_raise_invalid() -> None:
    with pytest.raises(
        NotImplementedError,
        match="Unable to infer datetime format, provided format is not supported.",
    ):
        parse_datetime_format(pa.chunked_array([["2024-01-01", "abc"]]))


@pytest.mark.parametrize(
    ("data", "duplicate"),
    [
        (["2024-01-01T00:00:00", "2024-01-01 01:00:00"], "separator"),
        (["2024-01-01 00:00:00+01:00", "2024-01-01 01:00:00+02:00"], "timezone"),
    ],
)
def test_pyarrow_infer_datetime_raise_not_unique(
    data: list[str | None], duplicate: str
) -> None:
    with pytest.raises(
        ValueError,
        match=f"Found multiple {duplicate} values while inferring datetime format.",
    ):
        parse_datetime_format(pa.chunked_array([data]))


@pytest.mark.parametrize("data", [["2024-01-01", "2024-12-01", "02-02-2024"]])
def test_pyarrow_infer_datetime_raise_inconsistent_date_fmt(
    data: list[str | None],
) -> None:
    with pytest.raises(ValueError, match="Unable to infer datetime format. "):
        parse_datetime_format(pa.chunked_array([data]))


@pytest.mark.parametrize("format", [None, "%Y-%m-%dT%H:%M:%S%z"])
def test_to_datetime_tz_aware(
    constructor: Constructor, request: pytest.FixtureRequest, format: str | None
) -> None:
    if is_pyarrow_windows_no_tzdata(constructor) or (
        "sqlframe" in str(constructor) and format is not None and is_windows()
    ):
        # NOTE: For `sqlframe` see https://github.com/narwhals-dev/narwhals/pull/2263#discussion_r2009101659
        pytest.skip()
    if "cudf" in str(constructor):
        # cuDF does not yet support timezone-aware datetimes
        request.applymarker(pytest.mark.xfail)
    context = (
        pytest.raises(NotImplementedError)
        if any(x in str(constructor) for x in ("duckdb", "ibis")) and format is None
        else does_not_raise()
    )
    df = nw.from_native(constructor({"a": ["2020-01-01T01:02:03+0100"]}))
    with context:
        result = df.with_columns(b=nw.col("a").str.to_datetime(format))
        assert isinstance(result.collect_schema()["b"], nw.Datetime)
        result_schema = result.lazy().collect().schema
        assert result_schema["a"] == nw.String
        assert isinstance(result_schema["b"], nw.Datetime)
        expected = {
            "a": ["2020-01-01T01:02:03+0100"],
            "b": [datetime(2020, 1, 1, 0, 2, 3, tzinfo=timezone.utc)],
        }
        assert_equal_data(result, expected)


@pytest.mark.skipif(PANDAS_VERSION < (2, 2, 0), reason="too old for pyarrow types")
def test_to_datetime_pd_preserves_pyarrow_backend_dtype() -> None:
    # Remark that pandas doesn't have a numpy-nullable datetime dtype, so
    # `.convert_dtypes(dtype_backend="numpy_nullable")` is a no-op in `_to_datetime(...)`
    pytest.importorskip("pandas")
    pytest.importorskip("pyarrow")
    import pandas as pd

    dtype_backend: Literal["pyarrow", "numpy_nullable"] = "pyarrow"

    df = nw.from_native(
        pd.DataFrame({"a": ["2020-01-01T12:34:56", None]}).convert_dtypes(
            dtype_backend=dtype_backend
        )
    )
    result = df.with_columns(b=nw.col("a").str.to_datetime()).to_native()
    result_dtype = get_dtype_backend(
        result["b"].dtype, df._compliant_frame._implementation
    )
    assert result_dtype == dtype_backend
