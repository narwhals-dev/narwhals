from __future__ import annotations

import re
from typing import TYPE_CHECKING, Any

import pytest

import narwhals as nw
from narwhals import _plan as nwp
from narwhals._plan import selectors as npcs
from narwhals.exceptions import InvalidOperationError
from tests.plan.utils import assert_equal_data, dataframe
from tests.utils import PYARROW_VERSION, assert_equal_data as _assert_equal_data

pytest.importorskip("pyarrow")


import pyarrow as pa

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from narwhals._plan.typing import IntoExpr

XFAIL_KEY_ERROR = pytest.mark.xfail(
    reason="TODO: Investigate 'Field * exists 2 times in schema'", raises=KeyError
)


def test_group_by_iter() -> None:
    data = {"a": [1, 1, 3], "b": [4, 4, 6], "c": [7.0, 8.0, 9.0]}
    df = dataframe(data)
    expected_keys: list[tuple[int, ...]] = [(1,), (3,)]
    keys = []
    for key, sub_df in df.group_by("a"):
        if key == (1,):
            expected = {"a": [1, 1], "b": [4, 4], "c": [7.0, 8.0]}
            assert_equal_data(sub_df, expected)
            assert isinstance(sub_df, nwp.DataFrame)
        keys.append(key)
    assert sorted(keys) == sorted(expected_keys)
    expected_keys = [(1, 4), (3, 6)]
    keys = [key for key, _ in df.group_by("a", "b")]
    assert sorted(keys) == sorted(expected_keys)
    keys = [key for key, _ in df.group_by("a", "b")]
    assert sorted(keys) == sorted(expected_keys)


def test_group_by_nw_all() -> None:
    df = dataframe({"a": [1, 1, 2], "b": [4, 5, 6], "c": [7, 8, 9]})
    result = df.group_by("a").agg(nwp.all().sum()).sort("a")
    expected = {"a": [1, 2], "b": [9, 6], "c": [15, 9]}
    assert_equal_data(result, expected)
    result = df.group_by("a").agg(nwp.all().sum().name.suffix("_sum")).sort("a")
    expected = {"a": [1, 2], "b_sum": [9, 6], "c_sum": [15, 9]}
    assert_equal_data(result, expected)


@pytest.mark.parametrize(
    ("attr", "expected"),
    [
        ("sum", {"a": [1, 2], "b": [3, 3]}),
        ("mean", {"a": [1, 2], "b": [1.5, 3]}),
        ("max", {"a": [1, 2], "b": [2, 3]}),
        ("min", {"a": [1, 2], "b": [1, 3]}),
        ("std", {"a": [1, 2], "b": [0.707107, None]}),
        ("var", {"a": [1, 2], "b": [0.5, None]}),
        ("len", {"a": [1, 2], "b": [3, 1]}),
        ("n_unique", {"a": [1, 2], "b": [3, 1]}),
        ("count", {"a": [1, 2], "b": [2, 1]}),
    ],
)
def test_group_by_depth_1_agg(attr: str, expected: dict[str, list[Any]]) -> None:
    data = {"a": [1, 1, 1, 2], "b": [1, None, 2, 3]}
    expr = getattr(nwp.col("b"), attr)()
    result = dataframe(data).group_by("a").agg(expr).sort("a")
    assert_equal_data(result, expected)


@pytest.mark.parametrize(
    ("values", "expected"),
    [
        (
            {"x": [True, True, True, False, False, False]},
            {"all": [True, False, False], "any": [True, True, False]},
        ),
        (
            {"x": [True, None, False, None, None, None]},
            {"all": [True, False, True], "any": [True, False, False]},
        ),
    ],
    ids=["not-nullable", "nullable"],
)
def test_group_by_depth_1_agg_bool_ops(
    values: dict[str, list[bool]], expected: dict[str, list[bool]]
) -> None:
    data = {"a": [1, 1, 2, 2, 3, 3], **values}
    result = (
        dataframe(data)
        .group_by("a")
        .agg(nwp.col("x").all().alias("all"), nwp.col("x").any().alias("any"))
        .sort("a")
    )
    assert_equal_data(result, {"a": [1, 2, 3], **expected})


@pytest.mark.parametrize(
    ("attr", "ddof"), [("std", 0), ("var", 0), ("std", 2), ("var", 2)]
)
def test_group_by_depth_1_std_var(attr: str, ddof: int) -> None:
    data = {"a": [1, 1, 1, 2, 2, 2], "b": [4, 5, 6, 0, 5, 5]}
    _pow = 0.5 if attr == "std" else 1
    expected = {
        "a": [1, 2],
        "b": [
            (sum((v - 5) ** 2 for v in [4, 5, 6]) / (3 - ddof)) ** _pow,
            (sum((v - 10 / 3) ** 2 for v in [0, 5, 5]) / (3 - ddof)) ** _pow,
        ],
    }
    expr = getattr(nwp.col("b"), attr)(ddof=ddof)
    result = dataframe(data).group_by("a").agg(expr).sort("a")
    assert_equal_data(result, expected)


def test_group_by_median() -> None:
    data = {"a": [1, 1, 1, 2, 2, 2], "b": [5, 4, 6, 7, 3, 2]}
    result = dataframe(data).group_by("a").agg(nwp.col("b").median()).sort("a")
    expected = {"a": [1, 2], "b": [5, 3]}
    assert_equal_data(result, expected)


def test_group_by_n_unique_w_missing() -> None:
    data = {"a": [1, 1, 2], "b": [4, None, 5], "c": [None, None, 7], "d": [1, 1, 3]}
    result = (
        dataframe(data)
        .group_by("a")
        .agg(
            nwp.col("b").n_unique(),
            c_n_unique=nwp.col("c").n_unique(),
            c_n_min=nwp.col("b").min(),
            d_n_unique=nwp.col("d").n_unique(),
        )
        .sort("a")
    )
    expected = {
        "a": [1, 2],
        "b": [2, 1],
        "c_n_unique": [1, 1],
        "c_n_min": [4, 5],
        "d_n_unique": [1, 1],
    }
    assert_equal_data(result, expected)


def test_group_by_simple_named() -> None:
    data = {"a": [1, 1, 2], "b": [4, 5, 6], "c": [7, 2, 1]}
    df = dataframe(data)
    result = (
        df.group_by("a").agg(b_min=nwp.col("b").min(), b_max=nwp.col("b").max()).sort("a")
    )
    expected = {"a": [1, 2], "b_min": [4, 6], "b_max": [5, 6]}
    assert_equal_data(result, expected)


def test_group_by_simple_unnamed() -> None:
    data = {"a": [1, 1, 2], "b": [4, 5, 6], "c": [7, 2, 1]}
    df = dataframe(data)
    result = df.group_by("a").agg(nwp.col("b").min(), nwp.col("c").max()).sort("a")
    expected = {"a": [1, 2], "b": [4, 6], "c": [7, 1]}
    assert_equal_data(result, expected)


def test_group_by_multiple_keys() -> None:
    data = {"a": [1, 1, 2], "b": [4, 4, 6], "c": [7, 2, 1]}
    df = dataframe(data)
    result = (
        df.group_by("a", "b")
        .agg(c_min=nwp.col("c").min(), c_max=nwp.col("c").max())
        .sort("a")
    )
    expected = {"a": [1, 2], "b": [4, 6], "c_min": [2, 1], "c_max": [7, 1]}
    assert_equal_data(result, expected)


def test_key_with_nulls() -> None:
    data = {"b": [4, 5, None], "a": [1, 2, 3]}
    result = (
        dataframe(data)
        .group_by("b")
        .agg(nwp.len(), nwp.col("a").min())
        .sort("a")
        .with_columns(nwp.col("b").cast(nw.Float64))
    )
    expected = {"b": [4.0, 5, None], "len": [1, 1, 1], "a": [1, 2, 3]}
    assert_equal_data(result, expected)


def test_key_with_nulls_ignored() -> None:
    data = {"b": [4, 5, None], "a": [1, 2, 3]}
    result = (
        dataframe(data)
        .group_by("b", drop_null_keys=True)
        .agg(nwp.len(), nwp.col("a").min())
        .sort("a")
        .with_columns(nwp.col("b").cast(nw.Float64))
    )
    expected = {"b": [4.0, 5], "len": [1, 1], "a": [1, 2]}
    assert_equal_data(result, expected)


def test_key_with_nulls_iter() -> None:
    data = {
        "b": [None, "4", "5", None, "7"],
        "a": [None, 1, 2, 3, 4],
        "c": [None, "4", "3", None, None],
    }
    result = dict(dataframe(data).group_by("b", "c", drop_null_keys=True).__iter__())

    assert len(result) == 2
    assert_equal_data(result[("4", "4")], {"b": ["4"], "a": [1], "c": ["4"]})
    assert_equal_data(result[("5", "3")], {"b": ["5"], "a": [2], "c": ["3"]})

    result = dict(dataframe(data).group_by("b", "c", drop_null_keys=False).__iter__())
    assert_equal_data(result[("4", "4")], {"b": ["4"], "a": [1], "c": ["4"]})
    assert_equal_data(result[("5", "3")], {"b": ["5"], "a": [2], "c": ["3"]})
    assert len(result) == 4


def test_group_by_expr_iter() -> None:
    data = {
        "b": [None, "4", "5", None, "7"],
        "a": [None, 1, 2, 3, 4],
        "c": ["1", "4", "3", "1", "1"],
    }

    expected = {
        ("1",): {"b": [None, None, "7"], "a": [None, 3, 4], "c": ["1", "1", "1"]},
        ("3",): {"b": ["5"], "a": [2], "c": ["3"]},
        ("4",): {"b": ["4"], "a": [1], "c": ["4"]},
    }
    grouped = dataframe(data).group_by(nwp.col("c").alias("d"))
    result = dict(sorted((k, df.sort("c").to_dict(as_series=False)) for k, df in grouped))
    assert len(result) == len(expected)
    assert result.keys() == expected.keys()
    # NOTE: The bug this is trying to avoid regressing on would break zipping, as one side has more columns
    result_p1 = next(iter(result.values()))
    expected_p1 = next(iter(expected.values()))
    assert result_p1 == expected_p1
    _assert_equal_data(result, expected)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "keys", [[nwp.col("a").abs()], ["a", nwp.col("a").abs().alias("a_test")]]
)
def test_group_by_raise_drop_null_keys_with_exprs(keys: list[nwp.Expr | str]) -> None:
    data = {"a": [1, 1, 2, 2, -1], "x": [0, 1, 2, 3, 4], "y": [0.5, -0.5, 1.0, -1.0, 1.5]}
    df = dataframe(data)
    with pytest.raises(
        NotImplementedError, match="drop_null_keys cannot be True when keys contains Expr"
    ):
        df.group_by(*keys, drop_null_keys=True).agg(nwp.sum("y"))  # type: ignore[call-overload]


def test_no_agg() -> None:
    data = {"a": [1, 1, 3], "b": [4, 4, 6], "c": [7.0, 8.0, 9.0]}
    result = dataframe(data).group_by(["a", "b"]).agg().sort("a", "b")
    expected = {"a": [1, 3], "b": [4, 6]}
    assert_equal_data(result, expected)


@pytest.mark.xfail(
    PYARROW_VERSION < (15,),
    reason=(
        "The defaults for grouping by categories in pandas are different.\n\n"
        "https://github.com/narwhals-dev/narwhals/issues/1078"
    ),
)
def test_group_by_categorical() -> None:
    data = {"g1": ["a", "a", "b", "b"], "g2": ["x", "y", "x", "z"], "x": [1, 2, 3, 4]}
    df = dataframe(data)
    result = (
        df.with_columns(
            g1=nwp.col("g1").cast(nw.Categorical()),
            g2=nwp.col("g2").cast(nw.Categorical()),
        )
        .group_by(["g1", "g2"])
        .agg(nwp.col("x").sum())
        .sort("x")
    )
    assert_equal_data(result, data)


@pytest.mark.parametrize(
    ("agg", "message_body", "expected_repr"),
    [
        (nwp.col("a").shift(1), r"shift.+not.+group_by.+pyarrow.+", "col('a').shift("),
        (
            nwp.col("a").arg_max(),
            r"arg_max.+not.+group_by.+pyarrow.+",
            "col('a').arg_max(",
        ),
        (
            nwp.col("a").max().over("b"),
            r"over.+not.+group_by.+pyarrow.+",
            "col('a').max().over([col('b')])",
        ),
        (
            nwp.col("a").drop_nulls().abs().mean(),
            r"complex aggregation found.+not.+group_by.+pyarrow.+",
            "col('a').drop_nulls().abs().mean()",
        ),
    ],
)
def test_group_by_unsupported_raises(
    agg: nwp.Expr, message_body: str, expected_repr: str
) -> None:
    df = dataframe({"a": [1, 2, 3], "b": [1, 1, 2]})
    pat = re.compile(rf"{message_body}{re.escape(expected_repr)}", re.DOTALL)
    with pytest.raises(InvalidOperationError, match=pat):
        df.group_by("b").agg(agg)


def test_double_same_aggregation() -> None:
    df = dataframe({"a": [1, 1, 2], "b": [4, 5, 6]})
    result = df.group_by("a").agg(c=nwp.col("b").mean(), d=nwp.col("b").mean()).sort("a")
    expected = {"a": [1, 2], "c": [4.5, 6], "d": [4.5, 6]}
    assert_equal_data(result, expected)


def test_all_kind_of_aggs() -> None:
    df = dataframe({"a": [1, 1, 1, 2, 2, 2], "b": [4, 5, 6, 0, 5, 5]})
    result = (
        df.group_by("a")
        .agg(
            c=nwp.col("b").mean(),
            d=nwp.col("b").mean(),
            e=nwp.col("b").std(ddof=1),
            f=nwp.col("b").std(ddof=2),
            g=nwp.col("b").var(ddof=2),
            h=nwp.col("b").var(ddof=2),
            i=nwp.col("b").n_unique(),
        )
        .sort("a")
    )

    variance_num = sum((v - 10 / 3) ** 2 for v in [0, 5, 5])
    expected = {
        "a": [1, 2],
        "c": [5, 10 / 3],
        "d": [5, 10 / 3],
        "e": [1, (variance_num / (3 - 1)) ** 0.5],
        "f": [2**0.5, (variance_num) ** 0.5],  # denominator is 1 (=3-2)
        "g": [2.0, variance_num],  # denominator is 1 (=3-2)
        "h": [2.0, variance_num],  # denominator is 1 (=3-2)
        "i": [3, 2],
    }
    assert_equal_data(result, expected)


@XFAIL_KEY_ERROR
def test_fancy_functions() -> None:
    df = dataframe({"a": [1, 1, 2], "b": [4, 5, 6]})
    result = df.group_by("a").agg(nwp.all().std(ddof=0)).sort("a")
    expected = {"a": [1, 2], "b": [0.5, 0.0]}
    assert_equal_data(result, expected)
    result = df.group_by("a").agg(npcs.numeric().std(ddof=0)).sort("a")
    assert_equal_data(result, expected)
    result = df.group_by("a").agg(npcs.matches("b").std(ddof=0)).sort("a")
    assert_equal_data(result, expected)
    result = df.group_by("a").agg(npcs.matches("b").std(ddof=0).alias("c")).sort("a")
    expected = {"a": [1, 2], "c": [0.5, 0.0]}
    assert_equal_data(result, expected)
    result = (
        df.group_by("a")
        .agg(npcs.matches("b").std(ddof=0).name.map(lambda _x: "c"))
        .sort("a")
    )
    assert_equal_data(result, expected)


@pytest.mark.parametrize(
    ("keys", "aggs", "expected", "sort_by"),
    [
        (
            [nwp.col("a").abs(), nwp.col("a").abs().alias("a_with_alias")],
            [nwp.col("x").sum()],
            {"a": [1, 2], "a_with_alias": [1, 2], "x": [5, 5]},
            ["a"],
        ),
        (
            [nwp.col("a").alias("x")],
            [nwp.col("x").mean().alias("y")],
            {"x": [-1, 1, 2], "y": [4.0, 0.5, 2.5]},
            ["x"],
        ),
        (
            [nwp.col("a")],
            [nwp.col("a").count().alias("foo-bar"), nwp.all().sum()],
            {"a": [-1, 1, 2], "foo-bar": [1, 2, 2], "x": [4, 1, 5], "y": [1.5, 0, 0]},
            ["a"],
        ),
        (
            [nwp.col("a", "y").abs()],
            [nwp.col("x").sum()],
            {"a": [1, 1, 2], "y": [0.5, 1.5, 1], "x": [1, 4, 5]},
            ["a", "y"],
        ),
        (
            [nwp.col("a").abs().alias("y")],
            [nwp.all().sum().name.suffix("c")],
            {"y": [1, 2], "ac": [1, 4], "xc": [5, 5]},
            ["y"],
        ),
        pytest.param(
            [npcs.by_dtype(nw.Float64()).abs()],
            [npcs.numeric().sum()],
            {"y": [0.5, 1.0, 1.5], "a": [2, 4, -1], "x": [1, 5, 4]},
            ["y"],
            marks=XFAIL_KEY_ERROR,
        ),
    ],
)
def test_group_by_expr(
    keys: list[nwp.Expr],
    aggs: list[nwp.Expr],
    expected: dict[str, list[Any]],
    sort_by: list[str],
) -> None:
    data = {"a": [1, 1, 2, 2, -1], "x": [0, 1, 2, 3, 4], "y": [0.5, -0.5, 1.0, -1.0, 1.5]}
    df = dataframe(data)
    result = df.group_by(*keys).agg(*aggs).sort(*sort_by)
    assert_equal_data(result, expected)


def test_group_by_expr_2757684799() -> None:
    """From [narwhals-dev/narwhals#2325-2757684799].

    The **incorrect** result is:

        {'b': [2, 1], 'a': [2, 1], 'c': [2.0, 1.0]}

    [narwhals-dev/narwhals#2325-2757684799]: https://github.com/narwhals-dev/narwhals/pull/2325#pullrequestreview-2757684799
    """
    data: dict[str, Any] = {"a": [1, 1, 2], "b": [4, 5, 6], "unrelated": [10, -1, -9]}
    df = dataframe(data)
    keys = nwp.col("a").alias("b"), "a"
    aggs = nwp.col("b").mean().alias("c")
    expected = {"b": [2, 1], "a": [2, 1], "c": [6.0, 4.5]}

    result = df.group_by(keys).agg(aggs).sort("b", descending=True)
    assert_equal_data(result, expected)


def test_group_by_selector() -> None:
    data = {
        "a": [1, 1, 1],
        "b": [4, 4, 6],
        "c": ["foo", "foo", "bar"],
        "x": [7.5, 8.5, 9.0],
    }
    result = (
        dataframe(data)
        .group_by(npcs.by_dtype(nw.Int64), "c")
        .agg(nwp.col("x").mean())
        .sort("a", "b")
    )
    expected = {"a": [1, 1], "b": [4, 6], "c": ["foo", "bar"], "x": [8.0, 9.0]}
    assert_equal_data(result, expected)


def test_renaming_edge_case() -> None:
    data = {"a": [0, 0, 0], "_a_tmp": [1, 2, 3], "b": [4, 5, 6]}
    result = dataframe(data).group_by(nwp.col("a")).agg(nwp.all().min())
    expected = {"a": [0], "_a_tmp": [1], "b": [4]}
    assert_equal_data(result, expected)


def test_group_by_len_1_column() -> None:
    """Based on a failure from marimo.

    - https://github.com/marimo-team/marimo/blob/036fd3ff89ef3a0e598bebb166637028024f98bc/tests/_plugins/ui/_impl/tables/test_narwhals.py#L1098-L1108
    - https://github.com/marimo-team/marimo/blob/036fd3ff89ef3a0e598bebb166637028024f98bc/marimo/_plugins/ui/_impl/tables/narwhals_table.py#L163-L188
    """
    data = {"a": [1, 2, 1, 2, 3, 4]}
    expected = {"a": [1, 2, 3, 4], "len": [2, 2, 1, 1], "len_a": [2, 2, 1, 1]}
    result = (
        dataframe(data).group_by("a").agg(nwp.len(), nwp.len().alias("len_a")).sort("a")
    )
    assert_equal_data(result, expected)


def test_top_level_len() -> None:
    # https://github.com/holoviz/holoviews/pull/6567#issuecomment-3178743331
    df = dataframe({"gender": ["m", "f", "f"], "weight": [4, 5, 6], "age": [None, 8, 9]})
    result = df.group_by(["gender"]).agg(nwp.all().len()).sort("gender")
    expected = {"gender": ["f", "m"], "weight": [2, 1], "age": [2, 1]}
    assert_equal_data(result, expected)
    result = (
        df.group_by("gender")
        .agg(nwp.col("weight").len(), nwp.col("age").len())
        .sort("gender")
    )
    assert_equal_data(result, expected)


@pytest.mark.parametrize(
    ("keys", "aggs", "expected", "pre_sort"),
    [
        (["a"], ["b"], {"a": [1, 2, 3, 4], "b": [1, 2, 4, 6]}, None),
        (["a"], ["b"], {"a": [1, 2, 3, 4], "b": [1, 3, 5, 6]}, {"descending": True}),
        (["a"], ["c"], {"a": [1, 2, 3, 4], "c": [None, "A", None, "B"]}, None),
        (
            ["a"],
            ["c"],
            {"a": [1, 2, 3, 4], "c": [None, "A", "B", "B"]},
            {"nulls_last": True},
        ),
    ],
    ids=["no-sort", "sort-descending", "NA-order-nulls-first", "NA-order-nulls-last"],
)
def test_group_by_agg_first(
    keys: Sequence[str],
    aggs: Sequence[str],
    expected: Mapping[str, Any],
    pre_sort: Mapping[str, Any] | None,
) -> None:
    data = {
        "a": [1, 2, 2, 3, 3, 4],
        "b": [1, 2, 3, 4, 5, 6],
        "c": [None, "A", "A", None, "B", "B"],
    }
    df = dataframe(data)
    if pre_sort:
        df = df.sort(aggs, **pre_sort)
    result = df.group_by(keys).agg(nwp.col(aggs).first()).sort(keys)
    assert_equal_data(result, expected)


@pytest.mark.parametrize(
    ("keys", "aggs", "expected", "pre_sort"),
    [
        (["a"], ["b"], {"a": [1, 2, 3, 4], "b": [1, 3, 5, 6]}, None),
        (["a"], ["b"], {"a": [1, 2, 3, 4], "b": [1, 2, 4, 6]}, {"descending": True}),
        (["a"], ["c"], {"a": [1, 2, 3, 4], "c": [None, "A", "B", "B"]}, None),
        (
            ["a"],
            ["c"],
            {"a": [1, 2, 3, 4], "c": [None, "A", None, "B"]},
            {"nulls_last": True},
        ),
    ],
    ids=["no-sort", "sort-descending", "NA-order-nulls-first", "NA-order-nulls-last"],
)
def test_group_by_agg_last(
    keys: Sequence[str],
    aggs: Sequence[str],
    expected: Mapping[str, Any],
    pre_sort: Mapping[str, Any] | None,
) -> None:
    data = {
        "a": [1, 2, 2, 3, 3, 4],
        "b": [1, 2, 3, 4, 5, 6],
        "c": [None, "A", "A", None, "B", "B"],
    }
    df = dataframe(data)
    if pre_sort:
        df = df.sort(aggs, **pre_sort)
    result = df.group_by(keys).agg(nwp.col(aggs).last()).sort(keys)
    assert_equal_data(result, expected)


@pytest.mark.parametrize(
    ("keys", "aggs", "expected"),
    [
        (["a"], [nwp.col("b").unique()], {"a": ["a", "b", "c"], "b": [[1], [2, 3], [3]]}),
        (
            ["a"],
            [nwp.col("b", "d").unique()],
            {
                "a": ["a", "b", "c"],
                "b": [[1], [2, 3], [3]],
                "d": [["three", "one"], ["three"], ["one"]],
            },
        ),
        pytest.param(
            ["d", "c"],
            [npcs.string().unique(), nwp.col("b").first().alias("b_first")],
            {
                "d": ["one", "one", "three", "three", "three"],
                "c": [1, 3, 2, 4, 5],
                "a": [["c"], ["a"], ["b"], ["b"], ["a"]],
                "b_first": [3, 1, 3, 2, 1],
            },
            marks=XFAIL_KEY_ERROR,
        ),
    ],
    ids=["Unique-Single", "Unique-Multi", "Unique-Selector-Fancy"],
)
def test_group_by_agg_unique(
    keys: Sequence[str], aggs: Sequence[IntoExpr], expected: Mapping[str, Any]
) -> None:
    data = {
        "a": ["a", "b", "a", "b", "c"],
        "b": [1, 2, 1, 3, 3],
        "c": [5, 4, 3, 2, 1],
        "d": ["three", "three", "one", "three", "one"],
    }
    df = dataframe(data)
    result = df.group_by(keys).agg(aggs).sort(keys)
    assert_equal_data(result, expected)


def test_group_by_args() -> None:
    """Adapted from [upstream].

    [upstream]: https://github.com/pola-rs/polars/blob/04dbc94c36f75ed05bb19587f2226e240ec1775f/py-polars/tests/unit/operations/test_group_by.py#L302-L325
    """
    data = {
        "a": ["a", "b", "a", "b", "b", "c"],
        "b": [1, 2, 3, 4, 5, 6],
        "c": [6, 5, 4, 3, 2, 1],
    }
    df = dataframe(data)

    # Single column name
    assert df.group_by("a").agg("b").columns == ["a", "b"]
    # Column names as list
    expected = ["a", "b", "c"]
    assert df.group_by(["a", "b"]).agg("c").columns == expected
    # Column names as positional arguments
    assert df.group_by("a", "b").agg("c").columns == expected
    # With keyword argument
    assert df.group_by("a", "b", drop_null_keys=True).agg("c").columns == expected
    # Multiple aggregations as list
    assert df.group_by("a").agg(["b", "c"]).columns == expected
    # Multiple aggregations as positional arguments
    assert df.group_by("a").agg("b", "c").columns == expected
    # Multiple aggregations as keyword arguments
    assert df.group_by("a").agg(q="b", r="c").columns == ["a", "q", "r"]


def test_group_by_all() -> None:
    """Adapted from [upstream].

    [upstream]: https://github.com/pola-rs/polars/blob/04dbc94c36f75ed05bb19587f2226e240ec1775f/py-polars/tests/unit/operations/test_group_by.py#L568-L577
    """
    data = {"a": [1, 2], "b": [1, 2]}
    df = dataframe(data)
    expected = {"a": [1, 2], "b": [1, 2], "a_agg": [1, 2]}
    result = df.group_by(nwp.all()).agg(nwp.col("a").max().name.suffix("_agg")).sort("a")
    assert_equal_data(result, expected)


def test_group_by_input_independent_with_len_23868() -> None:
    """Adapted from [upstream].

    [upstream]: https://github.com/pola-rs/polars/blob/04dbc94c36f75ed05bb19587f2226e240ec1775f/py-polars/tests/unit/operations/test_group_by.py#L1476-L1484
    """
    data = {"a": ["A", "B", "C"]}
    expected = {"literal": ["G"], "len": [3]}
    result = dataframe(data).group_by(nwp.lit("G")).agg(nwp.len())
    assert_equal_data(result, expected)


def test_group_by_series_lit_22103() -> None:
    """Adapted from [upstream], but rejecting for now.

    [upstream]: https://github.com/pola-rs/polars/blob/04dbc94c36f75ed05bb19587f2226e240ec1775f/py-polars/tests/unit/operations/test_group_by.py#L1406-L1424
    """
    data = {"g": [0, 1]}
    series = nwp.Series.from_native(pa.chunked_array([[42, 2, 3]]))
    df = dataframe(data)
    with pytest.raises(InvalidOperationError, match=re.escape("foo=lit(Series)")):
        df.group_by("g").agg(foo=series)


def test_group_by_named() -> None:
    """Adapted from [upstream].

    [upstream]: https://github.com/pola-rs/polars/blob/04dbc94c36f75ed05bb19587f2226e240ec1775f/py-polars/tests/unit/operations/test_group_by.py#L878-884
    """
    data = {"a": [1, 1, 2, 2, 3, 3], "b": range(6)}
    df = dataframe(data)
    result = df.group_by(z=nwp.col("a") * 2).agg(nwp.col("b").min()).sort("b")
    expected = (
        df.group_by((nwp.col("a") * 2).alias("z")).agg(nwp.col("b").min()).sort("b")
    )
    assert_equal_data(result, expected.to_dict(as_series=False))


def test_group_by_exclude_keys() -> None:
    # `group_by(keys)` and `exclude` share some logic
    data = {
        "a": ["A", "B", "A"],
        "b": [1, 2, 3],
        "c": [9, 2, 4],
        "d": [8, 7, 8],
        "e": [None, 9, 7],
        "f": [True, False, None],
        "g": [False, None, False],
        "h": [None, None, True],
        "j": [12.1, None, 4.0],
        "k": [42, 10, None],
        "l": [4, 5, 6],
        "m": [0, 1, 2],
    }
    df = dataframe(data).with_columns(
        npcs.boolean().fill_null(False), npcs.numeric().fill_null(0)
    )
    exclude = "b", "c", "d", "e", "f", "g", "j", "k", "l", "m"
    result = df.group_by(nwp.exclude(exclude)).agg(nwp.all().sum()).sort("a", "h")
    expected = {
        "a": ["A", "A", "B"],
        "h": [False, True, False],
        "b": [1, 3, 2],
        "c": [9, 4, 2],
        "d": [8, 8, 7],
        "e": [0, 7, 9],
        "f": [1, 0, 0],
        "g": [0, 0, 0],
        "j": [12.1, 4.0, 0.0],
        "k": [42, 0, 10],
        "l": [4, 6, 5],
        "m": [0, 2, 1],
    }
    assert_equal_data(result, expected)
