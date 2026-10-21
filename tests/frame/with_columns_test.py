from __future__ import annotations

import pytest
from packaging.version import Version as PkgVersion

import narwhals as nw
from narwhals.exceptions import ColumnNotFoundError, InvalidOperationError, ShapeError
from tests.utils import (
    POLARS_VERSION,
    PYARROW_VERSION,
    Constructor,
    ConstructorEager,
    assert_equal_data,
    maybe_collect,
)


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
    if "pyarrow_table" in str(constructor) and PYARROW_VERSION < PkgVersion("15"):
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


@pytest.mark.parametrize("from_dict", [False, True])
def test_with_columns_empty_series(
    constructor_eager: ConstructorEager, *, from_dict: bool
) -> None:
    df = nw.from_native(constructor_eager({"a": [0, 1, 2]}), eager_only=True)
    empty = nw.from_dict({}, backend=df.implementation) if from_dict else df.select()
    assert empty.shape == (0, 0)
    with pytest.raises(ShapeError):
        empty.with_columns(x=df["a"])


def test_with_columns_empty_selector_and_scalar(
    constructor_eager: ConstructorEager,
) -> None:
    # A selector that matches no columns adds nothing, so only the literal counts.
    df = nw.from_native(constructor_eager({"a": [1, 2, 3]}), eager_only=True)
    result = df.select().with_columns(nw.all(), nw.lit(1).alias("b"))
    assert result.shape == (0, 1)
    assert_equal_data(result, {"b": []})


def test_with_columns_empty_short_series(constructor_eager: ConstructorEager) -> None:
    df = nw.from_native(constructor_eager({"a": [1, 2, 3]}), eager_only=True)
    s = df["a"].head(1)
    # Polars 2 reports a length-1 Series on an empty frame as an invalid operation.
    with pytest.raises((ShapeError, InvalidOperationError)):
        df.select().with_columns(s)


@pytest.mark.parametrize(
    ("exprs", "columns"),
    [
        ([nw.lit(1)], ["literal"]),
        ([nw.lit(1), nw.lit(2).alias("second")], ["literal", "second"]),
    ],
)
def test_with_columns_empty_scalars(
    constructor_eager: ConstructorEager, exprs: list[nw.Expr], columns: list[str]
) -> None:
    df = nw.from_native(constructor_eager({"a": [0, 1, 2]}), eager_only=True).select()
    assert df.shape == (0, 0)
    result = df.with_columns(*exprs)
    assert result.shape == (0, len(columns))
    assert result.columns == columns
    assert_equal_data(result, {column: [] for column in columns})


def test_with_columns_empty_scalar_and_series(
    constructor_eager: ConstructorEager,
) -> None:
    df = nw.from_native(constructor_eager({"a": [0, 1, 2]}), eager_only=True)
    empty = df.select()
    assert empty.shape == (0, 0)
    with pytest.raises(ShapeError):
        empty.with_columns(nw.lit(1), x=df["a"])


def test_with_columns_zero_width_preserves_height(
    constructor_eager: ConstructorEager,
) -> None:
    if "polars" in str(constructor_eager) and POLARS_VERSION < PkgVersion("2"):
        # Polars before 2 cannot preserve height after dropping the last column.
        pytest.skip("Zero-width frames with positive height require Polars 2 or later")
    backend = nw.from_native(constructor_eager({"a": [0, 1, 2]})).implementation
    df = nw.from_dict({"a": [0, 1, 2]}, backend=backend)
    empty = df.drop("a")
    assert empty.shape == (3, 0)

    scalar_result = empty.with_columns(nw.lit(1))
    assert scalar_result.shape == (3, 1)
    assert_equal_data(scalar_result, {"literal": [1, 1, 1]})

    series_result = empty.with_columns(x=df["a"])
    assert series_result.shape == (3, 1)
    assert_equal_data(series_result, {"x": [0, 1, 2]})


@pytest.mark.parametrize("empty_on_left", [True, False])
def test_with_columns_empty_cross_join(
    constructor_eager: ConstructorEager, *, empty_on_left: bool
) -> None:
    df = nw.from_native(constructor_eager({"a": [0, 1, 2]}), eager_only=True)
    empty = df.select()
    result = empty.join(df, how="cross") if empty_on_left else df.join(empty, how="cross")
    assert result.shape == (0, 1)
    assert result.columns == ["a"]
    assert_equal_data(result, {"a": []})


def test_with_columns_zero_height(constructor_eager: ConstructorEager) -> None:
    df = nw.from_native(constructor_eager({"a": [0, 1, 2]}), eager_only=True)
    empty = df.head(0)
    assert empty.shape == (0, 1)
    with pytest.raises(ShapeError):
        empty.with_columns(x=df["a"])

    result = empty.with_columns(nw.lit(1))
    assert result.shape == (0, 2)
    assert_equal_data(result, {"a": [], "literal": []})
