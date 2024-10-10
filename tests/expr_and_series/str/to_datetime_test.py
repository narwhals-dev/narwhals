from typing import Any

import pytest

import narwhals.stable.v1 as nw
from tests.utils import Constructor

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


def test_to_datetime_series(constructor_eager: Any) -> None:
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


def test_to_datetime_infer_fmt(
    request: pytest.FixtureRequest, constructor: Constructor
) -> None:
    if "pyarrow_table" in str(constructor):
        request.applymarker(pytest.mark.xfail)

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


def test_to_datetime_series_infer_fmt(
    request: pytest.FixtureRequest, constructor_eager: Any
) -> None:
    if "pyarrow_table" in str(constructor_eager):
        request.applymarker(pytest.mark.xfail)

    if "cudf" in str(constructor_eager):  # pragma: no cover
        expected = "2020-01-01T12:34:56.000000000"
    else:
        expected = "2020-01-01 12:34:56"

    result = (
        nw.from_native(constructor_eager(data), eager_only=True)["a"].str.to_datetime()
    ).item(0)
    assert str(result) == expected
