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

# `a` is a non-list column (used to assert the dtype error); `l3`/`l4` are non-empty
# lists of matching per-row length, so they can be exploded together. The null/empty
# and flag behaviour lives in `test_explode_frame_options`.
data = {
    "a": ["x", "y", "z", "w"],
    "l3": [[1, 2], [3], [None], [1]],
    "l4": [[1, 2], [3], [123], [456]],
}


def _apply_explode_markers(
    request: pytest.FixtureRequest, constructor: Constructor, *, multi_col: bool = False
) -> None:
    unsupported = ["dask", "cudf", "pyarrow_table"]
    if multi_col:
        # Lazy SQL backends reject multi-column explode outright.
        unsupported.extend(["duckdb", "pyspark", "ibis"])
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
    # The null/empty + flag behaviour is covered by `test_explode_frame_options`.
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
    # `l3`/`l4` are all non-empty real lists of matching per-row length. The null/empty
    # + flag behaviour is covered by `test_explode_frame_options`.
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


# Ported from the Polars reference test (extended below with a `String`-typed list
# column `b`, so the explode path is exercised on more than just integer lists):
# https://github.com/pola-rs/polars/blob/1684cc09dfaa46656dfecc45ab866d01aa69bc78/py-polars/tests/unit/operations/test_explode.py#L596-L616
options_data = {
    "a": [[1, 2, 3], None, [4, 5, 6], []],
    "b": [[None, "dog", "cat"], None, ["narwhal", None, "orca"], []],
    "i": [1, 2, 3, 4],
}


@pytest.mark.parametrize(
    ("columns", "multi_col", "empty_as_null", "keep_nulls", "expected"),
    [
        # Single `Int32` list column `a` (column `b` is dropped via `select`).
        (
            ("a",),
            False,
            True,
            True,
            {"a": [1, 2, 3, None, 4, 5, 6, None], "i": [1, 1, 1, 2, 3, 3, 3, 4]},
        ),
        (
            ("a",),
            False,
            False,
            True,
            {"a": [1, 2, 3, None, 4, 5, 6], "i": [1, 1, 1, 2, 3, 3, 3]},
        ),
        (
            ("a",),
            False,
            True,
            False,
            {"a": [1, 2, 3, 4, 5, 6, None], "i": [1, 1, 1, 3, 3, 3, 4]},
        ),
        (("a",), False, False, False, {"a": [1, 2, 3, 4, 5, 6], "i": [1, 1, 1, 3, 3, 3]}),
        # `String` list column `b` exploded alongside `a`; both share the same per-row
        # null/empty layout, so the flags apply consistently across the two columns.
        (
            ("b", "a"),
            True,
            True,
            True,
            {
                "b": [None, "dog", "cat", None, "narwhal", None, "orca", None],
                "a": [1, 2, 3, None, 4, 5, 6, None],
                "i": [1, 1, 1, 2, 3, 3, 3, 4],
            },
        ),
        (
            ("b", "a"),
            True,
            False,
            True,
            {
                "b": [None, "dog", "cat", None, "narwhal", None, "orca"],
                "a": [1, 2, 3, None, 4, 5, 6],
                "i": [1, 1, 1, 2, 3, 3, 3],
            },
        ),
        (
            ("b", "a"),
            True,
            True,
            False,
            {
                "b": [None, "dog", "cat", "narwhal", None, "orca", None],
                "a": [1, 2, 3, 4, 5, 6, None],
                "i": [1, 1, 1, 3, 3, 3, 4],
            },
        ),
        (
            ("b", "a"),
            True,
            False,
            False,
            {
                "b": [None, "dog", "cat", "narwhal", None, "orca"],
                "a": [1, 2, 3, 4, 5, 6],
                "i": [1, 1, 1, 3, 3, 3],
            },
        ),
    ],
)
def test_explode_frame_options(
    request: pytest.FixtureRequest,
    constructor: Constructor,
    columns: tuple[str, ...],
    *,
    multi_col: bool,
    empty_as_null: bool,
    keep_nulls: bool,
    expected: dict[str, list[int | str | None]],
) -> None:
    _apply_explode_markers(request, constructor, multi_col=multi_col)

    result = (
        nw.from_native(constructor(options_data))
        .with_columns(
            nw.col("a").cast(nw.List(nw.Int32())), nw.col("b").cast(nw.List(nw.String()))
        )
        .explode(columns, empty_as_null=empty_as_null, keep_nulls=keep_nulls)
        .select(*columns, "i")
        # `i` is a stable per-source-row key and `a`'s values are unique within each
        # `i` group, so this fully determines row order for every backend.
        .sort("i", "a", nulls_last=True)
    )
    assert_equal_data(result, expected)


def test_explode_frame_single_elements(
    request: pytest.FixtureRequest, constructor: Constructor
) -> None:
    # Exploding a subset of the list columns leaves the others as-is, and the order in
    # which columns are passed to `explode` does not matter.
    # Ported from https://github.com/narwhals-dev/narwhals/pull/3347.
    _apply_explode_markers(request, constructor, multi_col=True)

    df = nw.from_native(
        constructor({"a": [[1], [2], [3]], "b": [[4], [5], [6]], "i": [0, 10, 20]})
    ).with_columns(nw.col("a", "b").cast(nw.List(nw.Int32())))

    single = df.explode("a").select("a", "b", "i").sort("i")
    assert_equal_data(single, {"a": [1, 2, 3], "b": [[4], [5], [6]], "i": [0, 10, 20]})

    both = df.explode("b", "a").select("a", "b", "i").sort("i")
    assert_equal_data(both, {"a": [1, 2, 3], "b": [4, 5, 6], "i": [0, 10, 20]})
