from __future__ import annotations

# ruff: noqa: FBT003
from typing import TYPE_CHECKING, Any

import pytest

pytest.importorskip("pyarrow")
pytest.importorskip("numpy")
import numpy as np
import pyarrow as pa

import narwhals as nw
from narwhals._plan import demo as nwd, selectors as ndcs
from narwhals._plan.common import is_expr
from narwhals._plan.dummy import DummyDataFrame
from narwhals._utils import Version
from narwhals.exceptions import ComputeError
from tests.utils import assert_equal_data

if TYPE_CHECKING:
    from collections.abc import Sequence

    from narwhals._plan.dummy import DummyExpr


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
def data_smaller(data_small: dict[str, Any]) -> dict[str, Any]:
    """Use only columns `"a"-"f"`."""
    keep = {"a", "b", "c", "d", "e", "f"}
    return {k: v for k, v in data_small.items() if k in keep}


def _ids_ir(expr: DummyExpr | Any) -> str:
    if is_expr(expr):
        return repr(expr._ir)
    return repr(expr)


XFAIL_REWRITE_SPECIAL_ALIASES = pytest.mark.xfail(
    reason="https://github.com/narwhals-dev/narwhals/blob/3732e5a6b56411157f13307dfdbd25e397d5b8e6/narwhals/_plan/meta.py#L142-L162\n"
    "Matches behavior of `polars`\n"
    "pl.select(pl.lit(1).name.suffix('_suffix'))",
    raises=ComputeError,
)
XFAIL_KLEENE_ALL_NULL = pytest.mark.xfail(
    reason="`pyarrow` uses `pa.null()`, which also fails in current `narwhals`.\n"
    "In `polars`, the same op is supported and it uses `pl.Null`.\n\n"
    "Function 'or_kleene' has no kernel matching input types (bool, null)",
    raises=pa.ArrowNotImplementedError,
)


@pytest.mark.parametrize(
    ("expr", "expected"),
    [
        (nwd.col("a"), {"a": ["A", "B", "A"]}),
        (nwd.col("a", "b"), {"a": ["A", "B", "A"], "b": [1, 2, 3]}),
        (nwd.lit(1), {"literal": [1]}),
        (nwd.lit(2.0), {"literal": [2.0]}),
        (nwd.lit(None, nw.String), {"literal": [None]}),
        (nwd.col("a", "b").first(), {"a": ["A"], "b": [1]}),
        (nwd.col("d").max(), {"d": [8]}),
        ([nwd.len(), nwd.nth(3).last()], {"len": [3], "d": [8]}),
        (
            [nwd.len().alias("e"), nwd.nth(3).last(), nwd.nth(2)],
            {"e": [3, 3, 3], "d": [8, 8, 8], "c": [9, 2, 4]},
        ),
        (nwd.col("b").sort(descending=True).alias("b_desc"), {"b_desc": [3, 2, 1]}),
        (nwd.col("c").filter(a="B"), {"c": [2]}),
        (
            [nwd.nth(0, 1).filter(nwd.col("c") >= 4), nwd.col("d").last() - 4],
            {"a": ["A", "A"], "b": [1, 3], "d": [4, 4]},
        ),
        (nwd.col("b").cast(nw.Float64()), {"b": [1.0, 2.0, 3.0]}),
        (nwd.lit(1).cast(nw.Float64).alias("literal_cast"), {"literal_cast": [1.0]}),
        pytest.param(
            nwd.lit(1).cast(nw.Float64()).name.suffix("_cast"),
            {"literal_cast": [1.0]},
            marks=XFAIL_REWRITE_SPECIAL_ALIASES,
        ),
        ([ndcs.string().first(), nwd.col("b")], {"a": ["A", "A", "A"], "b": [1, 2, 3]}),
        (
            nwd.col("c", "d")
            .sort_by("a", "b", descending=[True, False])
            .cast(nw.Float32())
            .name.to_uppercase(),
            {"C": [2.0, 9.0, 4.0], "D": [7.0, 8.0, 8.0]},
        ),
        ([nwd.int_range(5)], {"literal": [0, 1, 2, 3, 4]}),
        ([nwd.int_range(nwd.len())], {"literal": [0, 1, 2]}),
        (nwd.int_range(nwd.len() * 5, 20).alias("lol"), {"lol": [15, 16, 17, 18, 19]}),
        (nwd.int_range(nwd.col("b").min() + 4, nwd.col("d").last()), {"b": [5, 6, 7]}),
        (nwd.col("b") ** 2, {"b": [1, 4, 9]}),
        (
            [2 ** nwd.col("b"), (nwd.lit(2.0) ** nwd.nth(1)).alias("lit")],
            {"literal": [2, 4, 8], "lit": [2, 4, 8]},
        ),
        pytest.param(
            [
                nwd.col("b").is_between(2, 3, "left").alias("left"),
                nwd.col("b").is_between(2, 3, "right").alias("right"),
                nwd.col("b").is_between(2, 3, "none").alias("none"),
                nwd.col("b").is_between(2, 3, "both").alias("both"),
                nwd.col("c").is_between(
                    nwd.col("c").mean() - 1, 7 - nwd.col("b"), "both"
                ),
                nwd.col("c")
                .alias("c_right")
                .is_between(nwd.col("c").mean() - 1, 7 - nwd.col("b"), "right"),
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
                nwd.col("e").fill_null(0).alias("e_0"),
                nwd.col("e").fill_null(nwd.col("b")).alias("e_b"),
                nwd.col("e").fill_null(nwd.col("b").last()).alias("e_b_last"),
                nwd.col("e")
                .sort(nulls_last=True)
                .fill_null(nwd.col("d").last() - nwd.col("c"))
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
        (nwd.col("e", "d").is_null().any(), {"e": [True], "d": [False]}),
        (
            [(~nwd.col("e", "d").is_null()).all(), "b"],
            {"e": [False, False, False], "d": [True, True, True], "b": [1, 2, 3]},
        ),
        pytest.param(
            nwd.when(d=8).then("c"), {"c": [9, None, 4]}, id="When-otherwise-none"
        ),
        pytest.param(
            nwd.when(nwd.col("e").is_null())
            .then(nwd.col("b") + nwd.col("c"))
            .otherwise(50),
            {"b": [10, 50, 50]},
            id="When-otherwise-native-broadcast",
        ),
        pytest.param(
            nwd.when(nwd.col("a") == nwd.lit("C"))
            .then(nwd.lit("c"))
            .when(nwd.col("a") == nwd.lit("D"))
            .then(nwd.lit("d"))
            .when(nwd.col("a") == nwd.lit("B"))
            .then(nwd.lit("b"))
            .when(nwd.col("a") == nwd.lit("A"))
            .then(nwd.lit("a"))
            .alias("A"),
            {"A": ["a", "b", "a"]},
            id="When-then-x4",
        ),
        pytest.param(
            nwd.when(nwd.col("c") > 5, b=1).then(999),
            {"literal": [999, None, None]},
            id="When-multiple-predicates",
        ),
        pytest.param(
            nwd.when(nwd.col("b") == nwd.col("c"), nwd.col("d").mean() > nwd.col("d"))
            .then(123)
            .when(nwd.lit(True), ~nwd.nth(4).is_null())
            .then(456)
            .otherwise(nwd.col("c")),
            {"literal": [9, 123, 456]},
            id="When-multiple-predicates-mixed-broadcast",
        ),
        pytest.param(
            nwd.when(nwd.lit(True)).then("c"),
            {"c": [9, 2, 4]},
            id="When-literal-then-column",
        ),
        pytest.param(
            nwd.when(nwd.lit(True)).then(nwd.col("c").mean()),
            {"c": [5.0]},
            id="When-literal-then-agg",
        ),
        pytest.param(
            [
                nwd.when(nwd.lit(True)).then(nwd.col("e").last()),
                nwd.col("b").sort(descending=True),
            ],
            {"e": [7, 7, 7], "b": [3, 2, 1]},
            id="When-literal-then-agg-broadcast",
        ),
        pytest.param(
            [
                nwd.all_horizontal(
                    nwd.col("b") < nwd.col("c"),
                    nwd.col("a") != nwd.lit("B"),
                    nwd.col("e").cast(nw.Boolean),
                    nwd.lit(True),
                ),
                nwd.nth(1).last().name.suffix("_last"),
            ],
            {"b": [None, False, True], "b_last": [3, 3, 3]},
            id="all-horizontal-mixed-broadcast",
        ),
        pytest.param(
            [
                nwd.all_horizontal(nwd.lit(True), nwd.lit(True)).alias("a"),
                nwd.all_horizontal(nwd.lit(False), nwd.lit(True)).alias("b"),
                nwd.all_horizontal(nwd.lit(False), nwd.lit(False)).alias("c"),
                nwd.all_horizontal(nwd.lit(None, nw.Boolean), nwd.lit(True)).alias("d"),
                nwd.all_horizontal(nwd.lit(None, nw.Boolean), nwd.lit(False)).alias("e"),
                nwd.all_horizontal(
                    nwd.lit(None, nw.Boolean), nwd.lit(None, nw.Boolean)
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
                nwd.any_horizontal("f", "g"),
                nwd.any_horizontal("g", "h"),
                nwd.any_horizontal(nwd.lit(False), nwd.col("g").last()).alias(
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
                nwd.any_horizontal(nwd.lit(None, nw.Boolean), "i").alias("None-None"),
                nwd.any_horizontal(nwd.lit(True), "i").alias("True-None"),
                nwd.any_horizontal(nwd.lit(False), "i").alias("False-None"),
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
                nwd.col("b").alias("a"),
                nwd.col("l").alias("b"),
                nwd.col("m").alias("i"),
                nwd.any_horizontal(nwd.sum("b", "l").cast(nw.Boolean)).alias("any"),
                nwd.all_horizontal(nwd.sum("b", "l").cast(nw.Boolean)).alias("all"),
                nwd.max_horizontal(nwd.sum("b"), nwd.sum("l")).alias("max"),
                nwd.min_horizontal(nwd.sum("b"), nwd.sum("l")).alias("min"),
                nwd.sum_horizontal(nwd.sum("b"), nwd.sum("l")).alias("sum"),
                nwd.mean_horizontal(nwd.sum("b"), nwd.sum("l")).alias("mean"),
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
            nwd.mean_horizontal("j", nwd.col("k"), "e"),
            {"j": [27.05, 9.5, 5.5]},
            id="mean_horizontal-null",
        ),
        pytest.param(
            nwd.sum_horizontal("j", nwd.col("k"), "e"),
            {"j": [54.1, 19.0, 11.0]},
            id="sum_horizontal-null",
        ),
        pytest.param(
            nwd.concat_str(nwd.col("b") * 2, "n", nwd.col("o"), separator=" "),
            {"b": ["2 dogs play", "4 cats swim", None]},
            id="concat_str-preserve_nulls",
        ),
        pytest.param(
            nwd.concat_str(
                nwd.col("b") * 2, "n", nwd.col("o"), separator=" ", ignore_nulls=True
            ),
            {"b": ["2 dogs play", "4 cats swim", "6 walk"]},
            id="concat_str-ignore_nulls",
        ),
        pytest.param(
            nwd.concat_str("a", nwd.lit("a")),
            {"a": ["Aa", "Ba", "Aa"]},
            id="concat_str-lit",
        ),
        pytest.param(
            nwd.concat_str(
                nwd.lit("a"),
                nwd.lit("b"),
                nwd.lit("c"),
                nwd.lit("d"),
                nwd.col("e").last() + 13,
                separator="|",
            ),
            {"literal": ["a|b|c|d|20"]},
            id="concat_str-all-lit",
        ),
        pytest.param(
            [
                nwd.col("a")
                .alias("...")
                .map_batches(
                    lambda s: s.from_iterable(
                        [*((len(s) - 1) * [type(s.dtype).__name__.lower()]), "last"],
                        version=Version.MAIN,
                        name="funky",
                    ),
                    is_elementwise=True,
                ),
                nwd.col("a"),
            ],
            {"funky": ["string", "string", "last"], "a": ["A", "B", "A"]},
            id="map_batches-series",
        ),
        pytest.param(
            nwd.col("b")
            .map_batches(lambda s: s.to_numpy() + 1, nw.Float64(), is_elementwise=True)
            .sum(),
            {"b": [9.0]},
            id="map_batches-numpy",
        ),
        pytest.param(
            ndcs.by_name("b", "c", "d")
            .map_batches(lambda s: np.append(s.to_numpy(), [10, 2]), is_elementwise=True)
            .sort(),
            {"b": [1, 2, 2, 3, 10], "c": [2, 2, 4, 9, 10], "d": [2, 7, 8, 8, 10]},
            id="map_batches-selector",
        ),
        pytest.param(
            nwd.col("j", "k")
            .fill_null(15)
            .map_batches(lambda s: (s.to_numpy().max()), returns_scalar=True),
            {"j": [15], "k": [42]},
            id="map_batches-return_scalar",
            marks=pytest.mark.xfail(
                reason="not implemented `map_batches(returns_scalar=True)` for `pyarrow`",
                raises=NotImplementedError,
            ),
        ),
    ],
    ids=_ids_ir,
)
def test_select(
    expr: DummyExpr | Sequence[DummyExpr],
    expected: dict[str, Any],
    data_small: dict[str, Any],
) -> None:
    frame = pa.table(data_small)
    df = DummyDataFrame.from_native(frame)
    result = df.select(expr).to_dict(as_series=False)
    assert_equal_data(result, expected)


@pytest.mark.parametrize(
    ("expr", "expected"),
    [
        (
            ["d", nwd.col("a"), "b", nwd.col("e")],
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
            ndcs.numeric().cast(nw.String),
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
                nwd.col("e").fill_null(nwd.col("e").last()),
                nwd.col("f").sort(),
                nwd.nth(1).max(),
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
                nwd.col("a").alias("a?"),
                ndcs.by_name("a"),
                nwd.col("b").cast(nw.Float64).name.suffix("_float"),
                nwd.col("c").max() + 1,
                nwd.sum_horizontal(1, "d", nwd.col("b"), nwd.lit(3)),
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
    expr: DummyExpr | Sequence[DummyExpr],
    expected: dict[str, Any],
    data_smaller: dict[str, Any],
) -> None:
    frame = pa.table(data_smaller)
    df = DummyDataFrame.from_native(frame)
    result = df.with_columns(expr).to_dict(as_series=False)
    assert_equal_data(result, expected)


if TYPE_CHECKING:

    def test_protocol_expr() -> None:
        """Static test for all members implemented.

        There's a lot left to implement, but only gets detected if we invoke `__init__`, which
        doesn't happen elsewhere at the moment.
        """
        pytest.importorskip("pyarrow")
        from narwhals._plan.arrow.expr import ArrowExpr, ArrowScalar

        expr = ArrowExpr()
        scalar = ArrowScalar()
        assert expr
        assert scalar
