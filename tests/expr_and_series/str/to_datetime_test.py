from __future__ import annotations

from contextlib import nullcontext as does_not_raise
from datetime import datetime
from datetime import timezone
from typing import TYPE_CHECKING

import pyarrow as pa
import pytest

import narwhals.stable.v1 as nw
from narwhals._arrow.utils import parse_datetime_format
from tests.utils import assert_equal_data

if TYPE_CHECKING:
    from tests.utils import Constructor
    from tests.utils import ConstructorEager

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
    if "sqlframe" in str(constructor):
        # https://github.com/eakmanrq/sqlframe/issues/326
        assert result_schema["b"].time_zone == "UTC"
    else:
        assert result_schema["b"].time_zone is None
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
    if "duckdb" in str(constructor):
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
    constructor: Constructor,
    request: pytest.FixtureRequest,
    format: str | None,  # noqa: A002
) -> None:
    context = (
        pytest.raises(NotImplementedError)
        if any(x in str(constructor) for x in ("duckdb", "sqlframe")) and format is None
        else does_not_raise()
    )
    df = nw.from_native(constructor({"a": ["2020-01-01T01:02:03+0100"]}))
    with context:
        result = df.with_columns(b=nw.col("a").str.to_datetime(format))
        result_schema = result.collect_schema()
        assert result_schema["a"] == nw.String
        assert isinstance(result_schema["b"], nw.Datetime)
        if "polars_lazy" in str(constructor):
            # bug? report to Polars?
            assert result_schema["b"].time_zone is None
        else:
            assert result_schema["b"].time_zone == "UTC"
        if "sqlframe" in str(constructor):
            # https://github.com/eakmanrq/sqlframe/issues/325
            request.applymarker(pytest.mark.xfail)
        expected = {
            "a": ["2020-01-01T01:02:03+0100"],
            "b": [datetime(2020, 1, 1, 0, 2, 3, tzinfo=timezone.utc)],
        }
        assert_equal_data(result, expected)
