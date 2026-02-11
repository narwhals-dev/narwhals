from __future__ import annotations

from typing import Any

import pytest

import narwhals as nw
from narwhals.exceptions import ColumnNotFoundError, InvalidIntoExprError, NarwhalsError
from tests.utils import (
    DASK_VERSION,
    DUCKDB_VERSION,
    Constructor,
    ConstructorEager,
    assert_equal_data,
    maybe_collect,
)


class Foo: ...


def test_select(constructor: Constructor) -> None:
    data = {"a": [1, 3, 2], "b": [4, 4, 6], "z": [7.0, 8.0, 9.0]}
    df = nw.from_native(constructor(data))
    result = df.select("a")
    expected = {"a": [1, 3, 2]}
    assert_equal_data(result, expected)


def test_empty_select(constructor_eager: ConstructorEager) -> None:
    result = nw.from_native(constructor_eager({"a": [1, 2, 3]}), eager_only=True).select()
    assert result.shape == (0, 0)


def test_non_string_select() -> None:
    pytest.importorskip("pandas")
    import pandas as pd

    df = nw.from_native(pd.DataFrame({0: [1, 2], "b": [3, 4]}))
    result = nw.to_native(df.select(nw.col(0)))  # type: ignore[arg-type]
    expected = pd.Series([1, 2], name=0).to_frame()
    pd.testing.assert_frame_equal(result, expected)


def test_int_select_pandas() -> None:
    pytest.importorskip("pandas")
    import pandas as pd

    df = nw.from_native(pd.DataFrame({0: [1, 2], "b": [3, 4]}))
    with pytest.raises(InvalidIntoExprError, match="\n\nHint:\n- if you were trying"):
        nw.to_native(df.select(0))  # type: ignore[arg-type]


@pytest.mark.parametrize("invalid_select", [None, 0, Foo()])
def test_invalid_select(constructor: Constructor, invalid_select: Any) -> None:
    with pytest.raises(InvalidIntoExprError):
        nw.from_native(constructor({"a": [1, 2, 3]})).select(invalid_select)


def test_select_boolean_cols() -> None:
    pytest.importorskip("pandas")
    import pandas as pd

    df = nw.from_native(pd.DataFrame({True: [1, 2], False: [3, 4]}), eager_only=True)
    result = df.group_by(True).agg(nw.col(False).max())  # type: ignore[arg-type, call-overload]
    assert_equal_data(result.to_dict(as_series=False), {True: [1, 2]})  # type: ignore[dict-item]
    result = df.select(nw.col([False, True]))  # type: ignore[list-item]
    assert_equal_data(result.to_dict(as_series=False), {True: [1, 2], False: [3, 4]})  # type: ignore[dict-item]


def test_select_boolean_cols_multi_group_by() -> None:
    pytest.importorskip("pandas")
    import pandas as pd

    df = nw.from_native(
        pd.DataFrame({True: [1, 2], False: [3, 4], 2: [1, 1]}), eager_only=True
    )
    result = df.group_by(True, 2).agg(nw.col(False).max())  # type: ignore[arg-type, call-overload]
    assert_equal_data(
        result.to_dict(as_series=False),
        {True: [1, 2], 2: [1, 1], False: [3, 4]},  # type: ignore[dict-item]
    )

    result = df.select(nw.col([False, True]))  # type: ignore[list-item]
    assert_equal_data(result.to_dict(as_series=False), {True: [1, 2], False: [3, 4]})  # type: ignore[dict-item]


def test_comparison_with_list_error_message() -> None:
    pytest.importorskip("pyarrow")
    pytest.importorskip("pandas")
    import pandas as pd
    import pyarrow as pa

    msg = "Expected Series or scalar, got list."
    with pytest.raises(TypeError, match=msg):
        nw.from_native(pa.chunked_array([[1, 2, 3]]), series_only=True) == [1, 2, 3]  # noqa: B015
    with pytest.raises(TypeError, match=msg):
        nw.from_native(pd.Series([[1, 2, 3]]), series_only=True) == [1, 2, 3]  # noqa: B015


def test_missing_columns(
    constructor: Constructor, request: pytest.FixtureRequest
) -> None:
    constructor_id = str(request.node.callspec.id)
    if any(id_ == constructor_id for id_ in ("sqlframe", "ibis")):
        # `sqlframe` raises a different error depending on its underlying backend
        request.applymarker(pytest.mark.xfail)

    data = {"a": [1, 3, 2], "b": [4, 4, 6], "z": [7.0, 8.0, 9.0]}
    df = nw.from_native(constructor(data))
    selected_columns = ["a", "e", "f"]

    if constructor_id == "polars[lazy]":
        # In the lazy case, Polars only errors when we call `collect`,
        # and we have no way to recover exactly which columns the user
        # tried selecting. So, we just emit their message (which varies
        # across versions...)
        msg = r"e"
    elif constructor_id == "pyspark[connect]":  # pragma: no cover
        msg = r"^\[UNRESOLVED_COLUMN.WITH_SUGGESTION\]"
    elif any(id_ == constructor_id for id_ in ("duckdb", "pyspark")):
        msg = r"\n\nHint: Did you mean one of these columns: \['a', 'b', 'z'\]?"
    else:
        msg = (
            r"The following columns were not found: \[.*\]"
            r"\n\nHint: Did you mean one of these columns: \['a', 'b', 'z'\]?"
        )

    with pytest.raises(ColumnNotFoundError, match=msg):
        maybe_collect(df.select(selected_columns))

    # for the next two cases the error message is different in Polars
    if constructor_id == "polars[lazy]":
        msg = r"fdfa"
    elif constructor_id == "polars[eager]":
        msg = r"\n\nHint: Did you mean one of these columns: \['a', 'b', 'z'\]?"
    with pytest.raises(ColumnNotFoundError, match=msg):
        maybe_collect(df.select(nw.col("fdfa")))
    with pytest.raises(ColumnNotFoundError, match=msg):
        maybe_collect(df.select(nw.col("fdfa").sum()))


def test_left_to_right_broadcasting(constructor: Constructor) -> None:
    if "duckdb" in str(constructor) and DUCKDB_VERSION < (1, 3):
        pytest.skip()
    if "dask" in str(constructor) and DASK_VERSION < (2024, 10):
        pytest.skip()
    df = nw.from_native(constructor({"a": [1, 1, 2], "b": [4, 5, 6]}))
    result = df.select(nw.col("a") + nw.col("b").sum())
    expected = {"a": [16, 16, 17]}
    assert_equal_data(result, expected)
    result = df.select(nw.col("b").sum() + nw.col("a"))
    expected = {"b": [16, 16, 17]}
    assert_equal_data(result, expected)
    result = df.select(nw.col("b").sum() + nw.col("a").sum())
    expected = {"b": [19]}
    assert_equal_data(result, expected)


def test_alias_invalid(constructor: Constructor) -> None:
    df = nw.from_native(constructor({"a": [1, 2, 3], "b": [4, 5, 6]}))
    with pytest.raises((NarwhalsError, ValueError)):
        df.lazy().select(nw.all().alias("c")).collect()


def test_filtration_vs_aggregation(constructor_eager: ConstructorEager) -> None:
    df = nw.from_native(constructor_eager({"a": [1, None, 3]}))
    result = df.select(nw.col("a").drop_nulls(), b=nw.col("a").mean())
    expected: dict[str, Any] = {"a": [1, 3], "b": [2.0, 2.0]}
    assert_equal_data(result, expected)
    result = df.select(nw.sum_horizontal(nw.col("a").drop_nulls(), nw.col("a").mean()))
    expected = {"a": [3.0, 5.0]}
    assert_equal_data(result, expected)


def test_select_duplicates(constructor: Constructor) -> None:
    if "cudf" in str(constructor):
        # cudf already raises its own error
        pytest.skip()
    df = nw.from_native(constructor({"a": [1, 2]})).lazy()
    with pytest.raises(
        ValueError,
        match=r"Expected unique|[Dd]uplicate|more than one|Duplicate column name",
    ):
        df.select("a", nw.col("a") + 1).collect()


def test_binary_window_aggregation(constructor_eager: ConstructorEager) -> None:
    df = nw.from_native(constructor_eager({"a": [1, 1, 2]}))
    result = df.select(nw.col("a").cum_sum() + nw.col("a").sum())
    expected = {"a": [5, 6, 8]}
    assert_equal_data(result, expected)


def test_pandas_unnamed() -> None:
    pytest.importorskip("pandas")
    import pandas as pd

    df = nw.from_native(pd.DataFrame({None: [1, 2, 3]}), eager_only=True)
    result = df.select((nw.col(None) * 2).alias("foo"))  # type: ignore[arg-type]

    expected = {"foo": [2, 4, 6]}
    assert_equal_data(result, expected)
