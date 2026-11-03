from __future__ import annotations

import re
from typing import Any

import pytest

import narwhals as nw
from tests.utils import (
    Constructor,
    ConstructorEager,
    _to_comparable_list,
    assert_equal_data,
)

data = {"s": ["foo bar", "foo_bar", "foo_bar_baz", "foo,bar"]}


@pytest.mark.parametrize(
    ("by", "expected"),
    [
        ("_", {"s": [["foo bar"], ["foo", "bar"], ["foo", "bar", "baz"], ["foo,bar"]]}),
        (",", {"s": [["foo bar"], ["foo_bar"], ["foo_bar_baz"], ["foo", "bar"]]}),
    ],
)
def test_str_split(constructor: Constructor, by: str, expected: Any) -> None:
    if "cudf" not in str(constructor) and (
        constructor.__name__.startswith("pandas")
        and "pyarrow" not in constructor.__name__
    ):
        df = nw.from_native(constructor(data))
        msg = re.escape("This operation requires a pyarrow-backed series. ")
        with pytest.raises(TypeError, match=msg):
            df.select(nw.col("s").str.split(by=by))
        return
    df = nw.from_native(constructor(data))
    result_frame = df.select(nw.col("s").str.split(by=by))
    assert_equal_data(result_frame, expected)


@pytest.mark.parametrize(
    ("by", "expected"),
    [
        ("_", {"s": [["foo bar"], ["foo", "bar"], ["foo", "bar", "baz"], ["foo,bar"]]}),
        (",", {"s": [["foo bar"], ["foo_bar"], ["foo_bar_baz"], ["foo", "bar"]]}),
    ],
)
def test_str_split_series(
    constructor_eager: ConstructorEager, by: str, expected: Any
) -> None:
    if "cudf" not in str(constructor_eager) and (
        constructor_eager.__name__.startswith("pandas")
        and "pyarrow" not in constructor_eager.__name__
    ):
        df = nw.from_native(constructor_eager(data), eager_only=True)
        msg = re.escape("This operation requires a pyarrow-backed series. ")
        with pytest.raises(TypeError, match=msg):
            df["s"].str.split(by=by)
        return
    df = nw.from_native(constructor_eager(data), eager_only=True)
    result_series = df["s"].str.split(by=by)
    assert_equal_data({"s": result_series}, expected)


@pytest.mark.parametrize(
    "by",
    [
        pytest.param("||", id="double-pipe"),
        pytest.param("..", id="double-dot"),
        pytest.param("[.]", id="brackets"),
        pytest.param("\\", id="backslash"),
        pytest.param("::", id="double-colon"),
    ],
)
def test_str_split_literal(constructor: Constructor, by: str) -> None:
    if (
        constructor.__name__.startswith("pandas")
        and "pyarrow" not in constructor.__name__
    ):
        pytest.skip("String splitting requires pyarrow-backed pandas columns")
    data = {
        "s": [
            f"left{by}right",
            f"{by}left{by}",
            f"left{by}{by}right",
            f"左{by}右",
            "unchanged",
            "",
            None,
        ]
    }
    expected = {
        "s": [
            ["left", "right"],
            ["", "left", ""],
            ["left", "", "right"],
            ["左", "右"],
            ["unchanged"],
            [""],
            None,
        ]
    }
    df = nw.from_native(constructor(data))
    result = df.select(nw.col("s").str.split(by))
    if isinstance(result, nw.LazyFrame):
        result = result.collect()
    assert_equal_data(result, expected)
    expected_lengths = [2, 3, 3, 2, 1, 1, None]
    assert [
        len(value) if isinstance(value, list) else None
        for value in _to_comparable_list(result["s"])
    ] == expected_lengths
    if isinstance(df, nw.DataFrame):
        series = df["s"].str.split(by)
        assert_equal_data({"s": series}, expected)
        assert [
            len(value) if isinstance(value, list) else None
            for value in _to_comparable_list(series)
        ] == expected_lengths


@pytest.mark.parametrize("dialect", ["bigquery", "snowflake"])
@pytest.mark.parametrize("by", ["||", "[.]", "\\", " ", "-", "::"])
def test_str_split_sqlframe_literal_dialects(
    dialect: str, by: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    pytest.importorskip("sqlframe")
    from sqlframe.base.session import _BaseSession
    from sqlframe.bigquery import BigQuerySession
    from sqlframe.snowflake import SnowflakeSession
    from sqlglot import exp, parse_one

    session_types: dict[str, type[BigQuerySession | SnowflakeSession]] = {
        "bigquery": BigQuerySession,
        "snowflake": SnowflakeSession,
    }
    session_type = session_types[dialect]
    # Compile SQL without opening cloud connections or loading credentials.
    monkeypatch.setattr(_BaseSession, "_instance", None)
    monkeypatch.setattr(session_type, "__init__", _BaseSession.__init__)
    session = session_type.Builder().getOrCreate()
    native = session.createDataFrame([(f"left{by}right",)], schema=["s"])
    result = nw.from_native(native).select(nw.col("s").str.split(by))
    query = result.to_native().sql(dialect=dialect, pretty=False)
    split = parse_one(query, read=dialect).find(exp.Split)
    assert split is not None
    assert split.expression == exp.Literal.string(by)
