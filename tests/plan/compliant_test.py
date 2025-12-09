from __future__ import annotations

import re
from collections.abc import Iterable
from typing import TYPE_CHECKING, Any

import pytest

from narwhals._plan import selectors as ncs
from narwhals.exceptions import ColumnNotFoundError, InvalidOperationError

pytest.importorskip("pyarrow")
pytest.importorskip("numpy")
import datetime as dt

import pyarrow as pa

import narwhals as nw
from narwhals import _plan as nwp
from tests.plan.utils import (
    assert_equal_data,
    assert_equal_series,
    dataframe,
    first,
    last,
    series,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

    from narwhals._plan.typing import ColumnNameOrSelector, OneOrIterable
    from narwhals.typing import PythonLiteral
    from tests.conftest import Data


@pytest.fixture
def data_small() -> dict[str, Any]:
    return {
        "a": ["A", "B", "A"],
        "b": [1, 2, 3],
        "c": [9, 2, 4],
        "d": [8, 7, 8],
        "e": [None, 9, 7],
        "f": [True, False, None],
        "g": [False, None, False],
        "h": [None, None, True],
        "i": [None, None, None],
        "j": [12.1, None, 4.0],
        "k": [42, 10, None],
        "l": [4, 5, 6],
        "m": [0, 1, 2],
        "n": ["dogs", "cats", None],
        "o": ["play", "swim", "walk"],
    }


@pytest.fixture
def data_small_af(data_small: dict[str, Any]) -> dict[str, Any]:
    """Use only columns `"a"-"f"`."""
    keep = {"a", "b", "c", "d", "e", "f"}
    return {k: v for k, v in data_small.items() if k in keep}


@pytest.fixture
def data_small_dh(data_small: dict[str, Any]) -> dict[str, Any]:
    """Use only columns `"d"-"h"`."""
    keep = {"d", "e", "f", "g", "h"}
    return {k: v for k, v in data_small.items() if k in keep}


@pytest.fixture
def data_indexed() -> dict[str, Any]:
    """Used in https://github.com/narwhals-dev/narwhals/pull/2528."""
    return {
        "a": [8, 2, 1, None],
        "b": [58, 5, 6, 12],
        "c": [2.5, 1.0, 3.0, 0.9],
        "d": [2, 1, 4, 3],
        "idx": [0, 1, 2, 3],
    }


def _ids_ir(expr: nwp.Expr | Any) -> str:
    if isinstance(expr, nwp.Expr):
        return repr(expr._ir)
    return repr(expr)


XFAIL_KLEENE_ALL_NULL = pytest.mark.xfail(
    reason="`pyarrow` uses `pa.null()`, which also fails in current `narwhals`.\n"
    "In `polars`, the same op is supported and it uses `pl.Null`.\n\n"
    "Function 'or_kleene' has no kernel matching input types (bool, null)",
    raises=pa.ArrowNotImplementedError,
)


@pytest.mark.parametrize(
    ("expr", "expected"),
    [
        (nwp.col("a"), {"a": ["A", "B", "A"]}),
        (nwp.col("a", "b"), {"a": ["A", "B", "A"], "b": [1, 2, 3]}),
        (nwp.lit(1), {"literal": [1]}),
        (nwp.lit(2.0), {"literal": [2.0]}),
        (nwp.lit(None, nw.String), {"literal": [None]}),
        (nwp.col("a", "b").first(), {"a": ["A"], "b": [1]}),
        (nwp.col("d").max(), {"d": [8]}),
        ([nwp.len(), nwp.nth(3).last()], {"len": [3], "d": [8]}),
        (
            [nwp.len().alias("e"), nwp.nth(3).last(), nwp.nth(2)],
            {"e": [3, 3, 3], "d": [8, 8, 8], "c": [9, 2, 4]},
        ),
        (nwp.col("b").sort(descending=True).alias("b_desc"), {"b_desc": [3, 2, 1]}),
        (nwp.col("c").filter(a="B"), {"c": [2]}),
        (
            [nwp.nth(0, 1).filter(nwp.col("c") >= 4), nwp.col("d").last() - 4],
            {"a": ["A", "A"], "b": [1, 3], "d": [4, 4]},
        ),
        (nwp.col("b").cast(nw.Float64()), {"b": [1.0, 2.0, 3.0]}),
        (nwp.lit(1).cast(nw.Float64).alias("literal_cast"), {"literal_cast": [1.0]}),
        pytest.param(
            nwp.lit(1).cast(nw.Float64()).name.suffix("_cast"), {"literal_cast": [1.0]}
        ),
        ([ncs.string().first(), nwp.col("b")], {"a": ["A", "A", "A"], "b": [1, 2, 3]}),
        (
            nwp.col("c", "d")
            .sort_by("a", "b", descending=[True, False])
            .cast(nw.Float32())
            .name.to_uppercase(),
            {"C": [2.0, 9.0, 4.0], "D": [7.0, 8.0, 8.0]},
        ),
        ([nwp.int_range(5)], {"literal": [0, 1, 2, 3, 4]}),
        ([nwp.int_range(nwp.len())], {"literal": [0, 1, 2]}),
        (nwp.int_range(nwp.len() * 5, 20).alias("lol"), {"lol": [15, 16, 17, 18, 19]}),
        (nwp.int_range(nwp.col("b").min() + 4, nwp.col("d").last()), {"b": [5, 6, 7]}),
        (
            [
                nwp.date_range(
                    dt.date(2020, 1, 1),
                    dt.date(2020, 4, 30),
                    interval="25d",
                    closed="none",
                )
            ],
            {
                "literal": [
                    dt.date(2020, 1, 26),
                    dt.date(2020, 2, 20),
                    dt.date(2020, 3, 16),
                    dt.date(2020, 4, 10),
                ]
            },
        ),
        (
            (
                nwp.date_range(
                    dt.date(2021, 1, 30),
                    nwp.lit(18747, nw.Int32).cast(nw.Date),
                    interval="90d",
                    closed="left",
                ).alias("date_range_cast_expr"),
                {"date_range_cast_expr": [dt.date(2021, 1, 30)]},
            )
        ),
        (nwp.col("b") ** 2, {"b": [1, 4, 9]}),
        (
            [2 ** nwp.col("b"), (nwp.lit(2.0) ** nwp.nth(1)).alias("lit")],
            {"literal": [2, 4, 8], "lit": [2, 4, 8]},
        ),
        pytest.param(
            [
                nwp.col("b").is_between(2, 3, "left").alias("left"),
                nwp.col("b").is_between(2, 3, "right").alias("right"),
                nwp.col("b").is_between(2, 3, "none").alias("none"),
                nwp.col("b").is_between(2, 3, "both").alias("both"),
                nwp.col("c").is_between(
                    nwp.col("c").mean() - 1, 7 - nwp.col("b"), "both"
                ),
                nwp.col("c")
                .alias("c_right")
                .is_between(nwp.col("c").mean() - 1, 7 - nwp.col("b"), "right"),
            ],
            {
                "left": [False, True, False],
                "right": [False, False, True],
                "none": [False, False, False],
                "both": [False, True, True],
                "c": [False, False, True],
                "c_right": [False, False, False],
            },
            id="is_between",
        ),
        pytest.param(
            [
                nwp.col("e").fill_null(0).alias("e_0"),
                nwp.col("e").fill_null(nwp.col("b")).alias("e_b"),
                nwp.col("e").fill_null(nwp.col("b").last()).alias("e_b_last"),
                nwp.col("e")
                .sort(nulls_last=True)
                .fill_null(nwp.col("d").last() - nwp.col("c"))
                .alias("e_sort_wild"),
            ],
            {
                "e_0": [0, 9, 7],
                "e_b": [1, 9, 7],
                "e_b_last": [3, 9, 7],
                "e_sort_wild": [7, 9, 4],
            },
            id="sort",
        ),
        (nwp.col("e", "d").is_null().any(), {"e": [True], "d": [False]}),
        (
            [(nwp.col("e", "d").is_not_null()).all(), "b"],
            {"e": [False, False, False], "d": [True, True, True], "b": [1, 2, 3]},
        ),
        pytest.param(
            nwp.when(d=8).then("c"), {"c": [9, None, 4]}, id="When-otherwise-none"
        ),
        pytest.param(
            nwp.when(nwp.col("e").is_null())
            .then(nwp.col("b") + nwp.col("c"))
            .otherwise(50),
            {"b": [10, 50, 50]},
            id="When-otherwise-native-broadcast",
        ),
        pytest.param(
            nwp.when(nwp.col("a") == nwp.lit("C"))
            .then(nwp.lit("c"))
            .when(nwp.col("a") == nwp.lit("D"))
            .then(nwp.lit("d"))
            .when(nwp.col("a") == nwp.lit("B"))
            .then(nwp.lit("b"))
            .when(nwp.col("a") == nwp.lit("A"))
            .then(nwp.lit("a"))
            .alias("A"),
            {"A": ["a", "b", "a"]},
            id="When-then-x4",
        ),
        pytest.param(
            nwp.when(nwp.col("c") > 5, b=1).then(999),
            {"literal": [999, None, None]},
            id="When-multiple-predicates",
        ),
        pytest.param(
            nwp.when(nwp.col("b") == nwp.col("c"), nwp.col("d").mean() > nwp.col("d"))
            .then(123)
            .when(nwp.lit(True), ~nwp.nth(4).is_null())
            .then(456)
            .otherwise(nwp.col("c")),
            {"literal": [9, 123, 456]},
            id="When-multiple-predicates-mixed-broadcast",
        ),
        pytest.param(
            nwp.when(nwp.lit(True)).then("c"),
            {"c": [9, 2, 4]},
            id="When-literal-then-column",
        ),
        pytest.param(
            nwp.when(nwp.lit(True)).then(nwp.col("c").mean()),
            {"c": [5.0]},
            id="When-literal-then-agg",
        ),
        pytest.param(
            [
                nwp.when(nwp.lit(True)).then(nwp.col("e").last()),
                nwp.col("b").sort(descending=True),
            ],
            {"e": [7, 7, 7], "b": [3, 2, 1]},
            id="When-literal-then-agg-broadcast",
        ),
        pytest.param(
            [
                nwp.all_horizontal(
                    nwp.col("b") < nwp.col("c"),
                    nwp.col("a") != nwp.lit("B"),
                    nwp.col("e").cast(nw.Boolean),
                    nwp.lit(True),
                ),
                nwp.nth(1).last().name.suffix("_last"),
            ],
            {"b": [None, False, True], "b_last": [3, 3, 3]},
            id="all-horizontal-mixed-broadcast",
        ),
        pytest.param(
            [
                nwp.all_horizontal(nwp.lit(True), nwp.lit(True)).alias("a"),
                nwp.all_horizontal(nwp.lit(False), nwp.lit(True)).alias("b"),
                nwp.all_horizontal(nwp.lit(False), nwp.lit(False)).alias("c"),
                nwp.all_horizontal(nwp.lit(None, nw.Boolean), nwp.lit(True)).alias("d"),
                nwp.all_horizontal(nwp.lit(None, nw.Boolean), nwp.lit(False)).alias("e"),
                nwp.all_horizontal(
                    nwp.lit(None, nw.Boolean), nwp.lit(None, nw.Boolean)
                ).alias("f"),
            ],
            {
                "a": [True],
                "b": [False],
                "c": [False],
                "d": [None],
                "e": [False],
                "f": [None],
            },
            id="all-horizontal-kleene",
        ),
        pytest.param(
            [
                nwp.any_horizontal("f", "g"),
                nwp.any_horizontal("g", "h"),
                nwp.any_horizontal(nwp.lit(False), nwp.col("g").last()).alias(
                    "False-False"
                ),
            ],
            {
                "f": [True, None, None],
                "g": [None, None, True],
                "False-False": [False, False, False],
            },
            id="any-horizontal-kleene",
        ),
        pytest.param(
            [
                nwp.any_horizontal(nwp.lit(None, nw.Boolean), "i").alias("None-None"),
                nwp.any_horizontal(nwp.lit(True), "i").alias("True-None"),
                nwp.any_horizontal(nwp.lit(False), "i").alias("False-None"),
            ],
            {
                "None-None": [None, None, None],
                "True-None": [True, True, True],
                "False-None": [None, None, None],
            },
            id="any-horizontal-kleene-full-null",
            marks=XFAIL_KLEENE_ALL_NULL,
        ),
        pytest.param(
            [
                nwp.col("b").alias("a"),
                nwp.col("l").alias("b"),
                nwp.col("m").alias("i"),
                nwp.any_horizontal(nwp.sum("b", "l").cast(nw.Boolean)).alias("any"),
                nwp.all_horizontal(nwp.sum("b", "l").cast(nw.Boolean)).alias("all"),
                nwp.max_horizontal(nwp.sum("b"), nwp.sum("l")).alias("max"),
                nwp.min_horizontal(nwp.sum("b"), nwp.sum("l")).alias("min"),
                nwp.sum_horizontal(nwp.sum("b"), nwp.sum("l")).alias("sum"),
                nwp.mean_horizontal(nwp.sum("b"), nwp.sum("l")).alias("mean"),
            ],
            {
                "a": [1, 2, 3],
                "b": [4, 5, 6],
                "i": [0, 1, 2],
                "any": [True, True, True],
                "all": [True, True, True],
                "max": [15, 15, 15],
                "min": [6, 6, 6],
                "sum": [21, 21, 21],
                "mean": [10.5, 10.5, 10.5],
            },
            id="sumh_broadcasting",
        ),
        pytest.param(
            nwp.mean_horizontal("j", nwp.col("k"), "e"),
            {"j": [27.05, 9.5, 5.5]},
            id="mean_horizontal-null",
        ),
        pytest.param(
            nwp.sum_horizontal("j", nwp.col("k"), "e"),
            {"j": [54.1, 19.0, 11.0]},
            id="sum_horizontal-null",
        ),
        pytest.param(
            nwp.concat_str(nwp.col("b") * 2, "n", nwp.col("o"), separator=" "),
            {"b": ["2 dogs play", "4 cats swim", None]},
            id="concat_str-preserve_nulls",
        ),
        pytest.param(
            nwp.concat_str(
                nwp.col("b") * 2, "n", nwp.col("o"), separator=" ", ignore_nulls=True
            ),
            {"b": ["2 dogs play", "4 cats swim", "6 walk"]},
            id="concat_str-ignore_nulls",
        ),
        pytest.param(
            nwp.concat_str("a", nwp.lit("a")),
            {"a": ["Aa", "Ba", "Aa"]},
            id="concat_str-lit",
        ),
        pytest.param(
            nwp.concat_str(
                nwp.lit("a"),
                nwp.lit("b"),
                nwp.lit("c"),
                nwp.lit("d"),
                nwp.col("e").last() + 13,
                separator="|",
            ),
            {"literal": ["a|b|c|d|20"]},
            id="concat_str-all-lit",
        ),
        pytest.param(
            [nwp.col("g").len(), nwp.col("m").last(), nwp.col("h").count()],
            {"g": [3], "m": [2], "h": [1]},
            id="len-count-with-nulls",
        ),
    ],
    ids=_ids_ir,
)
def test_select(
    expr: nwp.Expr | Sequence[nwp.Expr],
    expected: dict[str, Any],
    data_small: dict[str, Any],
) -> None:
    result = dataframe(data_small).select(expr)
    assert_equal_data(result, expected)


@pytest.mark.parametrize(
    ("expr", "expected"),
    [
        (
            ["d", nwp.col("a"), "b", nwp.col("e")],
            {
                "a": ["A", "B", "A"],
                "b": [1, 2, 3],
                "c": [9, 2, 4],
                "d": [8, 7, 8],
                "e": [None, 9, 7],
                "f": [True, False, None],
            },
        ),
        (
            ncs.numeric().cast(nw.String),
            {
                "a": ["A", "B", "A"],
                "b": ["1", "2", "3"],
                "c": ["9", "2", "4"],
                "d": ["8", "7", "8"],
                "e": [None, "9", "7"],
                "f": [True, False, None],
            },
        ),
        (
            [
                nwp.col("e").fill_null(nwp.col("e").last()),
                nwp.col("f").sort(),
                nwp.nth(1).max(),
            ],
            {
                "a": ["A", "B", "A"],
                "b": [3, 3, 3],
                "c": [9, 2, 4],
                "d": [8, 7, 8],
                "e": [7, 9, 7],
                "f": [None, False, True],
            },
        ),
        pytest.param(
            [
                nwp.col("a").alias("a?"),
                ncs.by_name("a"),
                nwp.col("b").cast(nw.Float64).name.suffix("_float"),
                nwp.col("c").max() + 1,
                nwp.sum_horizontal(1, "d", nwp.col("b"), nwp.lit(3)),
            ],
            {
                "a": ["A", "B", "A"],
                "b": [1, 2, 3],
                "c": [10, 10, 10],
                "d": [8, 7, 8],
                "e": [None, 9, 7],
                "f": [True, False, None],
                "a?": ["A", "B", "A"],
                "b_float": [1.0, 2.0, 3.0],
                "literal": [13, 13, 15],
            },
            id="with_columns-extend",
        ),
    ],
)
def test_with_columns(
    expr: nwp.Expr | Sequence[nwp.Expr],
    expected: dict[str, Any],
    data_small_af: dict[str, Any],
) -> None:
    result = dataframe(data_small_af).with_columns(expr)
    assert_equal_data(result, expected)


@pytest.mark.parametrize(
    ("expr", "expected"),
    [
        (nwp.all().first(), {"a": 8, "b": 58, "c": 2.5, "d": 2, "idx": 0}),
        (ncs.numeric().null_count(), {"a": 1, "b": 0, "c": 0, "d": 0, "idx": 0}),
        (
            ncs.by_index(range(5)).cast(nw.Boolean).fill_null(False).all(),
            {"a": False, "b": True, "c": True, "d": True, "idx": False},
        ),
    ],
)
def test_with_columns_all_aggregates(
    data_indexed: dict[str, Any], expr: nwp.Expr, expected: dict[str, PythonLiteral]
) -> None:
    height = len(next(iter(data_indexed.values())))
    expected_full = {k: height * [v] for k, v in expected.items()}
    result = dataframe(data_indexed).with_columns(expr)
    assert_equal_data(result, expected_full)


@pytest.mark.parametrize(
    ("agg", "expected"),
    [
        (first("a"), 8),
        (first("b"), 58),
        (first("c"), 2.5),
        (last("a"), None),
        (last("b"), 12),
        (last("c"), 0.9),
    ],
)
def test_first_last_expr_with_columns(
    data_indexed: dict[str, Any], agg: nwp.Expr, expected: PythonLiteral
) -> None:
    """Related https://github.com/narwhals-dev/narwhals/pull/2528#discussion_r2225930065."""
    height = len(next(iter(data_indexed.values())))
    expected_full = {"result": height * [expected]}
    frame = dataframe(data_indexed)
    expr = agg.over(order_by="idx").alias("result")
    result = frame.with_columns(expr).select("result")
    assert_equal_data(result, expected_full)


@pytest.mark.parametrize(
    ("index", "expected"), [(3, (None, 12, 0.9, 3, 3)), (1, (2, 5, 1.0, 1, 1))]
)
def test_row_is_py_literal(
    data_indexed: dict[str, Any], index: int, expected: tuple[PythonLiteral, ...]
) -> None:
    frame = dataframe(data_indexed)
    result = frame.row(index)
    assert all(v is None or isinstance(v, (int, float)) for v in result)
    assert result == expected
    pytest.importorskip("polars")
    import polars as pl

    polars_result = pl.DataFrame(data_indexed).row(index)
    assert result == polars_result


@pytest.mark.parametrize(
    ("columns", "expected"),
    [
        ("a", ["b", "c"]),
        (["a"], ["b", "c"]),
        (ncs.first(), ["b", "c"]),
        ([ncs.first()], ["b", "c"]),
        (["a", "b"], ["c"]),
        (~ncs.last(), ["c"]),
        ([ncs.integer() | ncs.enum()], ["c"]),
        ([ncs.first(), "b"], ["c"]),
        (ncs.all(), []),
        ([], ["a", "b", "c"]),
        (ncs.struct(), ["a", "b", "c"]),
    ],
)
def test_drop(columns: OneOrIterable[ColumnNameOrSelector], expected: list[str]) -> None:
    data = {"a": [1, 3, 2], "b": [4, 4, 6], "c": [7.0, 8.0, 9.0]}
    df = dataframe(data)
    if isinstance(columns, (str, nwp.Selector, list)):
        assert df.drop(columns).collect_schema().names() == expected
    else:  # pragma: no cover
        ...
    if not isinstance(columns, str) and isinstance(columns, Iterable):
        assert df.drop(*columns).collect_schema().names() == expected


def test_drop_strict() -> None:
    data = {"a": [1, 3, 2], "b": [4, 4, 6]}
    df = dataframe(data)
    with pytest.raises(ColumnNotFoundError):
        df.drop("z")
    with pytest.raises(ColumnNotFoundError, match=re.escape("not found: ['z']")):
        df.drop(ncs.last(), "z")
    assert df.drop("z", strict=False).collect_schema().names() == ["a", "b"]
    assert df.drop(ncs.last(), "z", strict=False).collect_schema().names() == ["a"]


def test_drop_nulls(data_small_dh: Data) -> None:
    df = dataframe(data_small_dh)
    expected: Data = {"d": [], "e": [], "f": [], "g": [], "h": []}
    result = df.drop_nulls()
    assert_equal_data(result, expected)


def test_drop_nulls_invalid(data_small_dh: Data) -> None:
    df = dataframe(data_small_dh)
    with pytest.raises(TypeError, match=r"cannot turn.+int.+into a selector"):
        df.drop_nulls(123)  # type: ignore[arg-type]
    with pytest.raises(
        InvalidOperationError, match=r"cannot turn.+col\('a'\).first\(\).+into a selector"
    ):
        df.drop_nulls(nwp.col("a").first())  # type: ignore[arg-type]

    with pytest.raises(ColumnNotFoundError):
        df.drop_nulls(["j", "k"])

    with pytest.raises(ColumnNotFoundError):
        df.drop_nulls(ncs.by_name("j", "k"))

    with pytest.raises(ColumnNotFoundError):
        df.drop_nulls(ncs.by_index(-999))


DROP_ROW_1: Data = {
    "d": [7, 8],
    "e": [9, 7],
    "f": [False, None],
    "g": [None, False],
    "h": [None, True],
}
KEEP_ROW_3: Data = {"d": [8], "e": [7], "f": [None], "g": [False], "h": [True]}


@pytest.mark.parametrize(
    ("subset", "expected"),
    [
        ("e", DROP_ROW_1),
        (nwp.col("e"), DROP_ROW_1),
        (ncs.by_index(1), DROP_ROW_1),
        (ncs.integer(), DROP_ROW_1),
        ([ncs.numeric() | ~ncs.boolean()], DROP_ROW_1),
        (["g", "h"], KEEP_ROW_3),
        ([ncs.by_name("g", "h"), "d"], KEEP_ROW_3),
    ],
)
def test_drop_nulls_subset(
    data_small_dh: Data, subset: OneOrIterable[ColumnNameOrSelector], expected: Data
) -> None:
    df = dataframe(data_small_dh)
    result = df.drop_nulls(subset)
    assert_equal_data(result, expected)


def test_dataframe_to_polars() -> None:
    pytest.importorskip("polars")
    import polars as pl
    from polars.testing import assert_frame_equal as pl_assert_frame_equal

    data = {
        "a": [1, 3, 2],
        "b": [None, 4, 6],
        "c": ["A", "C", "B"],
        "d": ["D", None, "E"],
        "e": [1.9, 5.0, 3.3],
        "f": [1.5, 3.4, None],
    }
    expected = pl.DataFrame(data)
    result = dataframe(data).to_polars()
    pl_assert_frame_equal(result, expected)


@pytest.mark.parametrize(
    "values",
    [
        [1, 3, 2],
        [None, 4, 6],
        ["A", "C", "B"],
        ["D", None, "E"],
        [1.9, 5.0, 3.3],
        [1.5, 3.4, None],
    ],
)
def test_series_to_polars(values: Sequence[PythonLiteral]) -> None:
    pytest.importorskip("polars")
    import polars as pl
    from polars.testing import assert_series_equal as pl_assert_series_equal

    expected = pl.Series(values)
    result = series(values).to_polars()
    pl_assert_series_equal(result, expected)


def test_dataframe_iter_columns(data_small: Data) -> None:
    df = dataframe(data_small)
    result = df.from_dict({s.name: s for s in df.iter_columns()}).to_dict(as_series=False)
    assert_equal_data(df, result)


def test_dataframe_from_dict_misc(data_small: Data) -> None:
    pytest.importorskip("pyarrow")
    items = iter(data_small.items())
    name, values = next(items)
    mapping: dict[str, Any] = {
        name: nwp.Series.from_iterable(values, name=name, backend="pyarrow")
    }
    mapping.update(items)
    result = nwp.DataFrame.from_dict(mapping)
    assert_equal_data(result, data_small)

    with pytest.raises(TypeError, match=r"from_dict.+without.+backend"):
        nwp.DataFrame.from_dict(data_small)  # type: ignore[arg-type]


def test_dataframe_to_struct(data_small_af: Data) -> None:
    pytest.importorskip("pyarrow")

    schema = {
        "a": nw.String(),
        "b": nw.Int64(),
        "c": nw.Int64(),
        "d": nw.Int64(),
        "e": nw.Int64(),
        "f": nw.Boolean(),
    }

    df = dataframe(data_small_af).with_columns(
        nwp.col(name).cast(dtype) for name, dtype in schema.items()
    )
    result = df.to_struct("struct_series")
    result_dtype = result.dtype
    assert isinstance(result_dtype, nw.Struct)
    result_schema = dict(result_dtype.to_schema())
    assert result_schema == schema

    expected = [
        {"a": "A", "b": 1, "c": 9, "d": 8, "e": None, "f": True},
        {"a": "B", "b": 2, "c": 2, "d": 7, "e": 9, "f": False},
        {"a": "A", "b": 3, "c": 4, "d": 8, "e": 7, "f": None},
    ]
    assert_equal_series(result, expected, "struct_series")


# TODO @dangotbanned: Split this up
def test_series_misc() -> None:
    pytest.importorskip("pyarrow")

    values = [1.0, None, 7.1, float("nan"), 4.9, 12.0, 1.1, float("nan"), 0.2, None]
    name = "ser"
    ser = nwp.Series.from_iterable(values, name=name, dtype=nw.Float64, backend="pyarrow")
    assert ser.is_empty() is False
    assert ser.has_nulls()
    assert ser.null_count() == 2

    is_null = ser.is_null()
    is_nan = ser.is_nan()
    is_not_null = ser.is_not_null()
    is_not_nan = ser.is_not_nan()
    is_useful = ~(is_null | is_nan)

    assert is_useful.any()

    assert_equal_series(is_null, ~is_not_null)
    assert_equal_series(~is_null, is_not_null)
    assert_equal_series(is_nan, ~is_not_nan)
    assert_equal_series(~is_nan, is_not_nan)

    expected = [False, None, False, False, False, False, False, False, False, None]
    assert_equal_series(is_null & is_nan, expected, name)
    expected = [False, True, False, False, False, False, False, False, False, True]
    assert_equal_series(is_null, expected, name)
    expected = [True, False, True, False, True, True, True, False, True, False]
    assert_equal_series(is_not_nan & is_not_null, expected, name)

    assert ser.unique().drop_nans().drop_nulls().count() == 6
    assert len(list(ser)) == len(values)


def test_series_sort() -> None:
    ser = series([1.0, 7.1, None, 4.9])
    assert_equal_series(ser.sort(), [None, 1.0, 4.9, 7.1], "")
    assert_equal_series(ser.sort(nulls_last=True), [1.0, 4.9, 7.1, None], "")
    assert_equal_series(ser.sort(descending=True), [None, 7.1, 4.9, 1.0], "")
    assert_equal_series(
        ser.sort(descending=True, nulls_last=True), [7.1, 4.9, 1.0, None], ""
    )


def test_series_cast() -> None:
    pytest.importorskip("pyarrow")
    ser = nwp.int_range(10, step=2, eager="pyarrow", dtype=nw.Int64)
    assert ser.dtype == nw.Int64
    ser_float = ser.cast(nw.Float64)
    assert ser_float.dtype == nw.Float64
    assert ser.dtype == nw.Int64
    result = ser_float + 0.5
    expected = [0.5, 2.5, 4.5, 6.5, 8.5]
    assert_equal_series(result, expected, "literal")


if TYPE_CHECKING:
    from typing_extensions import assert_type

    def test_protocol_expr() -> None:
        """Static test for all members implemented.

        There's a lot left to implement, but only gets detected if we invoke `__init__`, which
        doesn't happen elsewhere at the moment.
        """
        pytest.importorskip("pyarrow")
        from narwhals._plan import arrow as _arrow

        # NOTE: Intentionally leaving `ewm_mean` without a `not_implemented()` for another test
        expr = _arrow.Expr()  # type: ignore[abstract]
        scalar = _arrow.Scalar()
        df = _arrow.DataFrame()
        ser = _arrow.Series()
        assert expr
        assert scalar
        assert df
        assert ser

    def test_dataframe_from_native_overloads() -> None:
        """Ensure we can reveal the `NativeSeries` **without** a dependency."""
        data: dict[str, Any] = {}
        native_good = pa.table(data)
        result_good = nwp.DataFrame.from_native(native_good)
        assert_type(result_good, "nwp.DataFrame[pa.Table, pa.ChunkedArray[Any]]")

        native_bad = native_good.to_batches()[0]
        nwp.DataFrame.from_native(native_bad)  # type: ignore[call-overload]
        assert_type(native_bad, "pa.RecordBatch")

    def test_int_range_overloads() -> None:
        series = nwp.int_range(50, eager="pyarrow")
        assert_type(series, "nwp.Series[pa.ChunkedArray[Any]]")
        native = series.to_native()
        assert_type(native, "pa.ChunkedArray[Any]")
        roundtrip = nwp.Series.from_native(native)
        assert_type(roundtrip, "nwp.Series[pa.ChunkedArray[Any]]")

    def test_date_range_overloads() -> None:
        series = nwp.date_range(dt.date(2000, 1, 1), dt.date(2002, 1, 1), eager="pyarrow")
        assert_type(series, "nwp.Series[pa.ChunkedArray[Any]]")
        native = series.to_native()
        assert_type(native, "pa.ChunkedArray[Any]")
        roundtrip = nwp.Series.from_native(native)
        assert_type(roundtrip, "nwp.Series[pa.ChunkedArray[Any]]")
