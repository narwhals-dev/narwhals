from __future__ import annotations

from typing import TYPE_CHECKING

import pyarrow as pa
import pytest

import narwhals.stable.v1 as nw
from narwhals._arrow.utils import parse_datetime_format

if TYPE_CHECKING:
    from tests.utils import Constructor
    from tests.utils import ConstructorEager

data = {"a": ["2020-01-01T12:34:56"]}


def test_to_datetime(constructor: Constructor) -> None:
    if "cudf" in str(constructor):  # pragma: no cover
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


def test_to_datetime_series(constructor_eager: ConstructorEager) -> None:
    if "cudf" in str(constructor_eager):  # pragma: no cover
        expected = "2020-01-01T12:34:56.000000000"
    else:
        expected = "2020-01-01 12:34:56"

    result = (
        nw.from_native(constructor_eager(data), eager_only=True)["a"].str.to_datetime(
            format="%Y-%m-%dT%H:%M:%S"
        )
    ).item(0)
    assert str(result) == expected


def test_to_datetime_infer_fmt(constructor: Constructor) -> None:
    if "cudf" in str(constructor):  # pragma: no cover
        expected = "2020-01-01T12:34:56.000000000"
    else:
        expected = "2020-01-01 12:34:56"

    result = (
        nw.from_native(constructor(data))
        .lazy()
        .select(b=nw.col("a").str.to_datetime())
        .collect()
        .item(row=0, column="b")
    )
    assert str(result) == expected


def test_to_datetime_series_infer_fmt(constructor_eager: ConstructorEager) -> None:
    if "cudf" in str(constructor_eager):  # pragma: no cover
        expected = "2020-01-01T12:34:56.000000000"
    else:
        expected = "2020-01-01 12:34:56"

    result = (
        nw.from_native(constructor_eager(data), eager_only=True)["a"].str.to_datetime()
    ).item(0)
    assert str(result) == expected


@pytest.mark.parametrize("data", [["2024-01-01", "abc"], ["2024-01-01", None]])
def test_pyarrow_infer_datetime_raise_invalid(data: list[str | None]) -> None:
    with pytest.raises(
        NotImplementedError,
        match="Unable to infer datetime format, provided format is not supported.",
    ):
        parse_datetime_format(pa.chunked_array([data]))


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
