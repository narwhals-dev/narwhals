from __future__ import annotations

import pytest

import narwhals as nw
from narwhals.exceptions import InvalidOperationError, ShapeError
from tests.utils import (
    PANDAS_VERSION,
    POLARS_VERSION,
    Constructor,
    ConstructorEager,
    assert_equal_data,
)

# For context, polars allows to explode multiple columns only if the columns
# have matching element counts, therefore, l1 and l2 but not l1 and l3 together.
data = {
    "a": ["x", "y", "z", "w"],
    "l1": [[1, 2], None, [None], []],
    "l2": [[3, None], None, [42], []],
    "l3": [[1, 2], [3], [None], [1]],
    "l4": [[1, 2], [3], [123], [456]],
}


def _apply_explode_markers(
    request: pytest.FixtureRequest, constructor: Constructor, *, multi_col: bool = False
) -> None:
    unsupported = ["dask", "cudf", "pyarrow_table"]
    if multi_col:
        # Lazy SQL backends reject multi-column explode outright.
        unsupported += ["duckdb", "pyspark", "ibis"]
    if any(backend in str(constructor) for backend in unsupported):
        request.applymarker(pytest.mark.xfail)
    if "pandas" in str(constructor):
        if PANDAS_VERSION < (2, 2):
            pytest.skip()
        pytest.importorskip("pyarrow")


def test_explode_single_col(
    request: pytest.FixtureRequest, constructor: Constructor
) -> None:
    # `l3` has no null/empty lists, so every row survives regardless of the flags.
    # The default behaviour over null/empty lists is covered by
    # `test_explode_empty_as_null_keep_nulls`.
    _apply_explode_markers(request, constructor)

    result = (
        nw.from_native(constructor(data))
        .with_columns(nw.col("l3").cast(nw.List(nw.Int32())))
        .explode("l3")
        .select("a", "l3")
        .sort("a", "l3", nulls_last=True)
    )
    expected = {"a": ["w", "x", "x", "y", "z"], "l3": [1, 1, 2, 3, None]}
    assert_equal_data(result, expected)


def test_explode_multiple_cols(
    request: pytest.FixtureRequest, constructor: Constructor
) -> None:
    # `l3`/`l4` are all non-empty real lists of matching per-row length. The default
    # behaviour over null/empty lists is covered by `test_explode_multiple_cols_flags`.
    _apply_explode_markers(request, constructor, multi_col=True)

    result = (
        nw.from_native(constructor(data))
        .with_columns(nw.col("l3", "l4").cast(nw.List(nw.Int32())))
        .explode("l3", "l4")
        .select("a", "l3", "l4")
        .sort("a", "l3", nulls_last=True)
    )
    expected = {
        "a": ["w", "x", "x", "y", "z"],
        "l3": [1, 1, 2, 3, None],
        "l4": [456, 1, 2, 3, 123],
    }
    assert_equal_data(result, expected)


@pytest.mark.parametrize(
    ("empty_as_null", "keep_nulls", "expected"),
    [
        # Use l2 column, hence:
        # x corresponds to [3, None]: non-empty (with null element),
        # y to null element, z to a non-empty list, w to an empty list
        (True, True, {"a": ["w", "x", "x", "y", "z"], "l2": [None, 3, None, None, 42]}),
        (True, False, {"a": ["w", "x", "x", "z"], "l2": [None, 3, None, 42]}),
        (False, True, {"a": ["x", "x", "y", "z"], "l2": [3, None, None, 42]}),
        (False, False, {"a": ["x", "x", "z"], "l2": [3, None, 42]}),
    ],
)
def test_explode_empty_as_null_keep_nulls(
    request: pytest.FixtureRequest,
    constructor: Constructor,
    *,
    empty_as_null: bool,
    keep_nulls: bool,
    expected: dict[str, list[int | None]],
) -> None:
    _apply_explode_markers(request, constructor)

    result = (
        nw.from_native(constructor(data))
        .with_columns(nw.col("l2").cast(nw.List(nw.Int32())))
        .explode("l2", empty_as_null=empty_as_null, keep_nulls=keep_nulls)
        .select("a", "l2")
        .sort("a", "l2", nulls_last=True)
    )
    assert_equal_data(result, expected)


@pytest.mark.parametrize(
    ("empty_as_null", "keep_nulls", "expected"),
    [
        # `l1`/`l2` have matching layout per row: x non-empty, y null, z [None], w empty.
        # The flags apply to both columns consistently
        (
            True,
            True,
            {
                "a": ["w", "x", "x", "y", "z"],
                "l1": [None, 1, 2, None, None],
                "l2": [None, 3, None, None, 42],
            },
        ),
        (
            True,
            False,
            {
                "a": ["w", "x", "x", "z"],
                "l1": [None, 1, 2, None],
                "l2": [None, 3, None, 42],
            },
        ),
        (
            False,
            True,
            {
                "a": ["x", "x", "y", "z"],
                "l1": [1, 2, None, None],
                "l2": [3, None, None, 42],
            },
        ),
        (False, False, {"a": ["x", "x", "z"], "l1": [1, 2, None], "l2": [3, None, 42]}),
    ],
)
def test_explode_multiple_cols_flags(
    request: pytest.FixtureRequest,
    constructor: Constructor,
    *,
    empty_as_null: bool,
    keep_nulls: bool,
    expected: dict[str, list[int | None]],
) -> None:
    _apply_explode_markers(request, constructor, multi_col=True)

    result = (
        nw.from_native(constructor(data))
        .with_columns(nw.col("l1", "l2").cast(nw.List(nw.Int32())))
        .explode("l1", "l2", empty_as_null=empty_as_null, keep_nulls=keep_nulls)
        .select("a", "l1", "l2")
        .sort("a", "l1", nulls_last=True)
    )
    assert_equal_data(result, expected)


@pytest.mark.parametrize(
    ("empty_as_null", "keep_nulls", "expected"),
    [
        # A `null` list and an empty `[]` list at the *same* row index. Both expand to
        # the same number of rows only when the flags treat them identically, i.e. when
        # `empty_as_null == keep_nulls`; otherwise it is a shape mismatch (tested below).
        (True, True, {"a": ["p", "q"], "l1": [None, None], "l2": [None, None]}),
        (False, False, {"a": [], "l1": [], "l2": []}),
    ],
)
def test_explode_multiple_cols_null_and_empty_same_index(
    request: pytest.FixtureRequest,
    constructor: Constructor,
    *,
    empty_as_null: bool,
    keep_nulls: bool,
    expected: dict[str, list[int | None]],
) -> None:
    _apply_explode_markers(request, constructor, multi_col=True)

    # Row p: (null, []). Row q: ([], null).
    mixed = {"a": ["p", "q"], "l1": [None, []], "l2": [[], None]}
    result = (
        nw.from_native(constructor(mixed))
        .with_columns(nw.col("l1", "l2").cast(nw.List(nw.Int32())))
        .explode("l1", "l2", empty_as_null=empty_as_null, keep_nulls=keep_nulls)
        .select("a", "l1", "l2")
        .sort("a", "l1", nulls_last=True)
    )
    assert_equal_data(result, expected)


@pytest.mark.parametrize(
    ("l1", "l2", "empty_as_null", "keep_nulls"),
    [
        # First row always matches ([0] vs [0]); the second row is the mismatch. The
        # leading real list keeps the column types inferable for every backend.
        # Two real lists of different length.
        ([[0], [1, 2]], [[0], [3]], True, True),
        # A real list against a null/empty: never compatible, regardless of flags.
        ([[0], [1]], [[0], None], True, True),
        ([[0], [1]], [[0], []], True, True),
        # null vs empty at the same index: a mismatch under "mixed" flags only, because
        # the two expand to a different number of rows (1 vs 0).
        ([[0], None], [[0], []], True, False),
        ([[0], None], [[0], []], False, True),
    ],
)
def test_explode_multiple_cols_shape_error(
    request: pytest.FixtureRequest,
    constructor: Constructor,
    l1: list[list[int] | None],
    l2: list[list[int] | None],
    *,
    empty_as_null: bool,
    keep_nulls: bool,
) -> None:
    # `duckdb`, `pyspark` and `ibis` reject multi-column explode with a
    # `NotImplementedError` mentioning "matching element counts", which the match
    # below accepts, so they are *not* xfailed here (hence `multi_col=False`). Only
    # backends that don't implement `explode` at all are.
    _apply_explode_markers(request, constructor)

    # The Polars<1.36 emulation decides row-fate from the first column only, so it
    # cannot raise on the flag-dependent `null`-vs-`[]` mismatch (it would need the
    # 1.36+ native path). Real-vs-emptyish / length mismatches still raise everywhere.
    flags_match = empty_as_null == keep_nulls
    if "polars" in str(constructor) and POLARS_VERSION < (1, 36) and not flags_match:
        request.applymarker(pytest.mark.xfail(reason="needs Polars>=1.36", strict=True))

    df = (
        nw.from_native(constructor({"a": ["g", "p"], "l1": l1, "l2": l2}))
        .with_columns(nw.col("l1", "l2").cast(nw.List(nw.Int32())))
        .lazy()
    )
    # Polars uses two phrasings: "matching element counts" for length/real-vs-emptyish
    # mismatches, and "doesn't have the same length" for the flag-induced ones.
    with pytest.raises(
        (ShapeError, NotImplementedError),
        match=r"matching element counts|doesn't have the same length",
    ):
        df.explode(
            "l1", "l2", empty_as_null=empty_as_null, keep_nulls=keep_nulls
        ).collect()


def test_explode_invalid_operation_error(
    request: pytest.FixtureRequest, constructor: Constructor
) -> None:
    if any(x in str(constructor) for x in ("pyarrow_table", "dask")):
        request.applymarker(pytest.mark.xfail)

    if "polars" in str(constructor) and POLARS_VERSION < (0, 20, 6):
        pytest.skip()

    with pytest.raises(
        InvalidOperationError, match="`explode` operation not supported for dtype"
    ):
        _ = nw.from_native(constructor(data)).lazy().explode("a").collect()


def test_explode_invalid_operation_error_eager(
    request: pytest.FixtureRequest, constructor_eager: ConstructorEager
) -> None:
    # Eager explode errors should surface as narwhals exceptions, not native ones
    # (e.g. Polars wraps via `catch_polars_exception`).
    if "pyarrow_table" in str(constructor_eager):
        request.applymarker(pytest.mark.xfail)

    if "polars" in str(constructor_eager) and POLARS_VERSION < (0, 20, 6):
        pytest.skip()

    with pytest.raises(
        InvalidOperationError, match="`explode` operation not supported for dtype"
    ):
        _ = nw.from_native(constructor_eager(data)).explode("a")
