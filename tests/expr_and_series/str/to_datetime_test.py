from __future__ import annotations

from datetime import datetime
from datetime import timezone
from typing import TYPE_CHECKING
from typing import Sequence

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
        .collect()
        .item(row=0, column="b")
    )
    assert str(result) == expected


@pytest.mark.parametrize(
    ("fmt", "data", "expected", "expected_polars_duckdb_pyspark"),
    [
        (
            "%Y-%m-%d %H:%M:%S%z",
            {"a": ["2020-01-01 12:34:56+02:00"]},
            "2020-01-01 12:34:56+02:00",
            "2020-01-01 10:34:56+00:00",
        )
    ],
)
def test_to_datetime_tz_aware(
    request: pytest.FixtureRequest,
    constructor: Constructor,
    fmt: str,
    data: dict[str, Sequence[str]],
    expected: str,
    expected_polars_duckdb_pyspark: str,
) -> None:
    constructor_str = str(constructor)
    if any(
        name in constructor_str for name in ("polars", "duckdb", "pyspark", "sqlframe")
    ):
        expected = expected_polars_duckdb_pyspark
    elif "pyarrow" in constructor_str and "pandas" not in constructor_str:
        from pyarrow.lib import ArrowInvalid

        expected = expected_polars_duckdb_pyspark
        request.applymarker(
            pytest.mark.xfail(
                True,  # noqa: FBT003
                raises=ArrowInvalid,
                reason="Unclear, see https://github.com/narwhals-dev/narwhals/pull/2152#discussion_r1983225794",
            )
        )

    result = (
        nw.from_native(constructor(data))
        .lazy()
        .select(b=nw.col("a").str.to_datetime(fmt))
        .collect()
        .item(row=0, column="b")
    )

    assert str(result) == expected


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
