from __future__ import annotations

from typing import TYPE_CHECKING, cast

import pytest

import narwhals as nw
from tests.utils import ConstructorEager, assert_equal_data

if TYPE_CHECKING:
    from collections.abc import Callable
    from typing import Literal, TypeAlias

    from narwhals._compliant.typing import EagerSeriesAny
    from narwhals.typing import NonNestedLiteral

    Data: TypeAlias = "list[NonNestedLiteral]"
    Op: TypeAlias = "Callable[[EagerSeriesAny], EagerSeriesAny]"
    CompliantSeries: TypeAlias = "Callable[[Data], EagerSeriesAny]"
    Operand: TypeAlias = Literal["self", "mask", "other"]
    Case: TypeAlias = "tuple[Data, Op]"


def like(series: EagerSeriesAny, data: Data) -> EagerSeriesAny:
    return series.from_iterable(data * len(series), context=series)


PRESERVING: dict[str, Case] = {
    "__invert__": ([True], lambda s: ~s),
    "__neg__": ([1], lambda s: -s),
    "abs": ([-1], lambda s: s.abs()),
    "alias": ([1], lambda s: s.alias("b")),
    "cast": ([1], lambda s: s.cast(nw.Float64())),
    "ceil": ([1.2], lambda s: s.ceil()),
    "clip": ([5], lambda s: s.clip(like(s, [1]), like(s, [9]))),
    "clip_lower": ([5], lambda s: s.clip_lower(like(s, [1]))),
    "clip_upper": ([5], lambda s: s.clip_upper(like(s, [9]))),
    "cos": ([1.0], lambda s: s.cos()),
    "exp": ([1.0], lambda s: s.exp()),
    "fill_nan": ([float("nan")], lambda s: s.fill_nan(0.0)),
    "fill_null": ([1], lambda s: s.fill_null(like(s, [0]), None, None)),
    "floor": ([1.2], lambda s: s.floor()),
    "is_between": ([5], lambda s: s.is_between(like(s, [1]), like(s, [9]), "both")),
    "is_finite": ([1.0], lambda s: s.is_finite()),
    "is_in": ([1], lambda s: s.is_in([1, 2])),
    "is_nan": ([1.0], lambda s: s.is_nan()),
    "is_null": ([1], lambda s: s.is_null()),
    "log": ([8.0], lambda s: s.log(2.0)),
    "replace_strict": (
        [1],
        lambda s: s.replace_strict(like(s, [0]), [1], [2], return_dtype=None),
    ),
    "round": ([1.234], lambda s: s.round(2)),
    "sin": ([1.0], lambda s: s.sin()),
    "sqrt": ([4.0], lambda s: s.sqrt()),
    "zip_with": (["a"], lambda s: s.zip_with(like(s, [True]), like(s, ["b"]))),
}

LOSES_LENGTH_1: dict[str, Case] = {
    "filter": ([1], lambda s: s.filter(like(s, [False]))),
    "drop_nulls": ([None], lambda s: s.drop_nulls()),
    "head": ([1], lambda s: s.head(0)),
    "_align_full_broadcast": (
        [1],
        lambda s: s._align_full_broadcast(s, like(s, [1, 2, 3]))[0],
    ),
}

ZIP_WITH_EXPECTED: dict[Operand, list[str]] = {
    "self": ["a", "y", "a"],
    "mask": ["a", "b", "c"],
    "other": ["a", "x", "c"],
}


@pytest.fixture
def compliant_series(constructor_eager: ConstructorEager) -> CompliantSeries:
    if "polars" in str(constructor_eager):
        pytest.skip("polars series have no `_broadcast` flag")

    def compliant_series(data: Data) -> EagerSeriesAny:
        series = nw.from_native(constructor_eager({"a": data}), eager_only=True)["a"]
        return cast("EagerSeriesAny", series._compliant_series)

    return compliant_series


@pytest.mark.parametrize(("data", "op"), PRESERVING.values(), ids=PRESERVING)
def test_preserves_broadcast(
    compliant_series: CompliantSeries, data: Data, op: Op
) -> None:
    series = compliant_series(data)
    series._broadcast = True
    assert op(series)._broadcast is True


@pytest.mark.parametrize(("data", "op"), LOSES_LENGTH_1.values(), ids=LOSES_LENGTH_1)
def test_drops_broadcast(compliant_series: CompliantSeries, data: Data, op: Op) -> None:
    series = compliant_series(data)
    series._broadcast = True
    result = op(series)
    assert len(result) != 1
    assert result._broadcast is False


@pytest.mark.parametrize(("data", "op"), PRESERVING.values(), ids=PRESERVING)
def test_never_gains_broadcast(
    compliant_series: CompliantSeries, data: Data, op: Op
) -> None:
    series = compliant_series(data * 3)
    assert series._broadcast is False
    assert op(series)._broadcast is False


@pytest.mark.parametrize("broadcast", ["self", "mask", "other"])
def test_zip_with_broadcast_operand(
    compliant_series: CompliantSeries, broadcast: Operand
) -> None:
    operands: dict[Operand, EagerSeriesAny] = {
        "self": compliant_series(["a", "b", "c"]),
        "mask": compliant_series([True, False, True]),
        "other": compliant_series(["x", "y", "z"]),
    }
    operands[broadcast] = compliant_series(operands[broadcast].to_list()[:1])
    operands[broadcast]._broadcast = True
    result = operands["self"].zip_with(operands["mask"], operands["other"])
    assert result.to_list() == ZIP_WITH_EXPECTED[broadcast]
    assert result._broadcast is False


@pytest.mark.parametrize(
    ("expr", "expected"),
    [
        (nw.lit(1) + nw.col("a"), [2]),
        (nw.col("a").sum() + nw.col("a"), [2]),
        (nw.lit(5).clip(nw.col("a"), nw.col("a") + 10), [5]),
        (nw.when(nw.col("a") > 0).then(nw.lit(7)).otherwise(nw.col("a")), [7]),
        (nw.sum_horizontal(nw.lit(1), nw.col("a")), [2]),
        (nw.concat_str(nw.lit("!"), nw.col("s")), ["!x"]),
    ],
)
def test_scalar_left_operand_single_row(
    constructor_eager: ConstructorEager, expr: nw.Expr, expected: list[int | str]
) -> None:
    df = nw.from_native(constructor_eager({"a": [1], "s": ["x"]}))
    assert_equal_data(df.select(b=expr), {"b": expected})
