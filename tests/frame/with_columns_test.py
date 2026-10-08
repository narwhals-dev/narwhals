from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

import narwhals as nw
from narwhals.exceptions import ColumnNotFoundError, ShapeError
from tests.utils import (
    PYARROW_VERSION,
    Constructor,
    ConstructorEager,
    assert_equal_data,
    maybe_collect,
)

if TYPE_CHECKING:
    from narwhals._typing import EagerAllowed


def test_with_columns_int_col_name_pandas() -> None:
    pytest.importorskip("pandas")
    import numpy as np
    import pandas as pd

    np_matrix = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    df = pd.DataFrame(np_matrix, dtype="int64")
    nw_df = nw.from_native(df, eager_only=True)
    result = nw_df.with_columns(nw_df.get_column(1).alias(4)).pipe(nw.to_native)  # type: ignore[arg-type]
    expected = pd.DataFrame(
        {0: [1, 4, 7], 1: [2, 5, 8], 2: [3, 6, 9], 4: [2, 5, 8]}, dtype="int64"
    )
    pd.testing.assert_frame_equal(result, expected)


def test_with_columns_order(constructor: Constructor) -> None:
    data = {"a": [1, 3, 2], "b": [4, 4, 6], "z": [7.0, 8.0, 9.0]}
    df = nw.from_native(constructor(data))
    result = df.with_columns(nw.col("a") + 1, d=nw.col("a") - 1)
    assert result.collect_schema().names() == ["a", "b", "z", "d"]
    expected = {"a": [2, 4, 3], "b": [4, 4, 6], "z": [7.0, 8.0, 9.0], "d": [0, 2, 1]}
    assert_equal_data(result, expected)


def test_with_columns_empty(constructor_eager: ConstructorEager) -> None:
    data = {"a": [1, 3, 2], "b": [4, 4, 6], "z": [7.0, 8.0, 9.0]}
    df = nw.from_native(constructor_eager(data))
    result = df.select().with_columns()
    assert_equal_data(result, {})


def test_select_with_columns_empty_lazy(constructor: Constructor) -> None:
    data = {"a": [1, 3, 2], "b": [4, 4, 6], "z": [7.0, 8.0, 9.0]}
    df = nw.from_native(constructor(data)).lazy()
    with pytest.raises(ValueError, match="At least one"):
        df.with_columns()
    with pytest.raises(ValueError, match="At least one"):
        df.select()


def test_with_columns_order_single_row(constructor: Constructor) -> None:
    data = {"a": [1, 3, 2], "b": [4, 4, 6], "z": [7.0, 8.0, 9.0], "i": [0, 1, 2]}
    df = nw.from_native(constructor(data)).filter(nw.col("i") < 1).drop("i")
    result = df.with_columns(nw.col("a") + 1, d=nw.col("a") - 1)
    assert result.collect_schema().names() == ["a", "b", "z", "d"]
    expected = {"a": [2], "b": [4], "z": [7.0], "d": [0]}
    assert_equal_data(result, expected)


def test_with_columns_dtypes_single_row(
    constructor: Constructor, request: pytest.FixtureRequest
) -> None:
    if "pyarrow_table" in str(constructor) and PYARROW_VERSION < (15,):
        pytest.skip()
    if (
        ("pyspark" in str(constructor))
        or "duckdb" in str(constructor)
        or "ibis" in str(constructor)
    ):
        request.applymarker(pytest.mark.xfail)
    data = {"a": ["foo"]}
    df = nw.from_native(constructor(data)).with_columns(nw.col("a").cast(nw.Categorical))
    result = df.with_columns(nw.col("a"))
    assert result.collect_schema() == {"a": nw.Categorical}


def test_with_columns_series_shape_mismatch(constructor_eager: ConstructorEager) -> None:
    df1 = nw.from_native(constructor_eager({"first": [1, 2, 3]}), eager_only=True)
    second = nw.from_native(constructor_eager({"second": [1, 2, 3, 4]}), eager_only=True)[
        "second"
    ]
    with pytest.raises(ShapeError):
        df1.with_columns(second=second)


def test_with_columns_missing_column(
    constructor: Constructor, request: pytest.FixtureRequest
) -> None:
    constructor_id = str(request.node.callspec.id)
    if any(id_ == constructor_id for id_ in ("sqlframe", "ibis")):
        # `sqlframe` raises a different error depending on its underlying backend
        request.applymarker(pytest.mark.xfail)
    data = {"a": [1, 2], "b": [3, 4]}
    df = nw.from_native(constructor(data))

    if "polars" in str(constructor):
        msg = r"c"
    elif any(id_ == constructor_id for id_ in ("duckdb", "pyspark")):
        msg = r"\n\nHint: Did you mean one of these columns: \['a', 'b'\]?"
    elif constructor_id == "pyspark[connect]":  # pragma: no cover
        msg = r"^\[UNRESOLVED_COLUMN.WITH_SUGGESTION\]"
    else:
        msg = (
            r"The following columns were not found: \[.*\]"
            r"\n\nHint: Did you mean one of these columns: \['a', 'b'\]?"
        )

    with pytest.raises(ColumnNotFoundError, match=msg):
        maybe_collect(df.with_columns(d=nw.col("c") + 1))


def test_with_columns_series_on_zero_column_frame(
    constructor_eager: ConstructorEager,
) -> None:
    source = nw.from_native(constructor_eager({"a": [0, 1, 2, 3, 4]}), eager_only=True)
    result = source.select().with_columns(additional_column=source["a"])
    assert_equal_data(result, {"additional_column": [0, 1, 2, 3, 4]})


@pytest.mark.parametrize("from_numpy", [False, True])
def test_with_columns_series_on_empty_input(
    eager_backend: EagerAllowed, *, from_numpy: bool
) -> None:
    import numpy as np

    source = nw.from_dict({"a": [0, 1, 2, 3, 4]}, backend=eager_backend)
    empty = (
        nw.from_numpy(np.empty((0, 0)), backend=eager_backend)
        if from_numpy
        else nw.from_dict({}, backend=eager_backend)
    )
    result = empty.with_columns(additional_column=source["a"])
    assert_equal_data(result, {"additional_column": [0, 1, 2, 3, 4]})
    assert empty.shape == (0, 0)


def test_with_columns_literals_on_zero_column_frame(
    constructor_eager: ConstructorEager,
) -> None:
    empty = nw.from_native(constructor_eager({"a": [0]}), eager_only=True).select()
    assert_equal_data(empty.with_columns(nw.lit(1)), {"literal": [1]})
    result = empty.with_columns(nw.lit(1), additional_column=nw.lit(2))
    assert_equal_data(result, {"literal": [1], "additional_column": [2]})
    assert empty.shape == (0, 0)


@pytest.mark.parametrize("literal_first", [False, True])
def test_with_columns_literal_and_series_on_zero_column_frame(
    constructor_eager: ConstructorEager, *, literal_first: bool
) -> None:
    source = nw.from_native(constructor_eager({"a": [0, 1, 2]}), eager_only=True)
    empty = source.select()
    literal = nw.lit(1).alias("literal")
    result = (
        empty.with_columns(literal, source["a"])
        if literal_first
        else empty.with_columns(source["a"], literal)
    )
    expected = {"literal": [1, 1, 1], "a": [0, 1, 2]}
    assert_equal_data(
        result, expected if literal_first else dict(reversed(expected.items()))
    )


def test_with_columns_mismatched_series_on_zero_column_frame(
    constructor_eager: ConstructorEager,
) -> None:
    source = nw.from_native(constructor_eager({"a": [0, 1, 2]}), eager_only=True)
    shorter = source.head(2)["a"]
    with pytest.raises(ShapeError):
        source.select().with_columns(source["a"], shorter=shorter)


def test_with_columns_on_zero_row_frame_with_columns(
    constructor_eager: ConstructorEager,
) -> None:
    source = nw.from_native(constructor_eager({"a": [0, 1, 2, 3, 4]}), eager_only=True)
    empty = source.head(0)
    assert_equal_data(empty.with_columns(), {"a": []})
    with pytest.raises(ShapeError):
        empty.with_columns(additional_column=source["a"])
    assert_equal_data(empty.with_columns(nw.lit(1)), {"a": [], "literal": []})


def test_with_columns_on_zero_column_frame_with_index() -> None:
    pd = pytest.importorskip("pandas")
    native = pd.DataFrame(index=[4, 5, 6])
    empty = nw.from_native(native, eager_only=True)
    series = nw.from_native(pd.Series([0, 1, 2]), series_only=True)
    result = empty.with_columns(additional_column=series, literal=nw.lit(1))
    assert_equal_data(result, {"additional_column": [0, 1, 2], "literal": [1, 1, 1]})
    assert result.to_native().index.tolist() == [4, 5, 6]
    assert native.shape == (3, 0)
