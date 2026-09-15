from __future__ import annotations

from typing import TYPE_CHECKING, cast

import pytest

import narwhals as nw
from narwhals.exceptions import ShapeError
from tests.utils import POLARS_VERSION, ConstructorEager, assert_equal_data

if TYPE_CHECKING:
    from collections.abc import Callable
    from typing import Literal, TypeAlias

    from narwhals._compliant.typing import EagerSeriesAny
    from narwhals.typing import NonNestedLiteral

    Data: TypeAlias = "list[NonNestedLiteral]"
    Op: TypeAlias = "Callable[[EagerSeriesAny], EagerSeriesAny]"
    CompliantSeries: TypeAlias = "Callable[[Data], EagerSeriesAny]"
    Operand: TypeAlias = Literal["self", "mask", "other"]


# `_broadcast` is set once, by `EagerExpr.broadcast`, on the final series of a
# scalar-like expression. No Series method carries it: anything that combines
# such series by hand aligns them first with `_align_full_broadcast`.
NEVER_CARRIED: dict[str, Op] = {
    "abs": lambda s: s.abs(),
    "cast": lambda s: s.cast(nw.Float64()),
    "is_null": lambda s: s.is_null(),
    "fill_null": lambda s: s.fill_null(s, None, None),
    "__add__": lambda s: s + s,
}

LENGTH_CHANGING: dict[str, nw.Expr] = {
    "add": nw.lit(10) + nw.col("a").unique(),
    "radd": nw.col("a").unique() + nw.lit(10),
    "clip": nw.lit(10).clip(nw.col("a").unique(), nw.lit(99)),
    "is_between": nw.lit(10).is_between(nw.col("a").unique(), nw.lit(99), "both"),
    "when_then": nw.when(nw.col("a").unique() > 0).then(nw.lit(1)).otherwise(nw.lit(2)),
}

# Polars only rejects a non-scalar unit-length column on its in-memory engine, its
# streaming engine broadcasts instead (pola-rs/polars#26023, still open). Which engine
# runs eager `DataFrame` methods has moved around, so the expectation is banded:
#   <= 1.9        no `ShapeError` (1.7 to 1.9 raise `InvalidOperationError` instead)
#   1.10 to 1.24  raises
#   1.25 to 1.43  broadcasts
#   >= 1.44       raises, eager pinned to the in-memory engine (pola-rs/polars#28800)
POLARS_EAGER_RAISES = (1, 10) <= POLARS_VERSION < (1, 25) or POLARS_VERSION >= (1, 44)
# `when/then` reached the in-memory engine's check only in 1.44.
POLARS_WHEN_RAISES = POLARS_VERSION >= (1, 44)

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


@pytest.mark.parametrize("op", NEVER_CARRIED.values(), ids=NEVER_CARRIED)
def test_series_ops_never_carry_broadcast(
    compliant_series: CompliantSeries, op: Op
) -> None:
    series = compliant_series([1])
    series._broadcast = True
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
        (nw.col("a") + nw.lit(1), [2]),
        (nw.col("a").sum() + nw.col("a"), [2]),
        (nw.col("a") + nw.col("a").sum(), [2]),
        (nw.lit(5).clip(nw.col("a"), nw.col("a") + 10), [5]),
        (nw.col("a").clip(nw.lit(5), nw.lit(10)), [5]),
        (nw.when(nw.col("a") > 0).then(nw.lit(7)).otherwise(nw.col("a")), [7]),
        (nw.sum_horizontal(nw.lit(1), nw.col("a")), [2]),
        (nw.mean_horizontal(nw.lit(1), nw.col("a")), [1.0]),
        (nw.min_horizontal(nw.lit(4), nw.col("a")), [1]),
        (nw.max_horizontal(nw.lit(4), nw.col("a")), [4]),
        (nw.all_horizontal(nw.lit(True), nw.col("a") > 0, ignore_nulls=True), [True]),
        (nw.any_horizontal(nw.lit(False), nw.col("a") > 0, ignore_nulls=True), [True]),
        (nw.concat_str(nw.lit("!"), nw.col("s")), ["!x"]),
        (nw.concat_str(nw.col("s"), nw.lit("!")), ["x!"]),
    ],
)
def test_scalar_operand_single_row(
    constructor_eager: ConstructorEager, expr: nw.Expr, expected: list[object]
) -> None:
    df = nw.from_native(constructor_eager({"a": [1], "s": ["x"]}))
    assert_equal_data(df.select(b=expr), {"b": expected})


@pytest.mark.parametrize("name", LENGTH_CHANGING)
def test_length_changing_operand_is_not_scalar(
    request: pytest.FixtureRequest, constructor_eager: ConstructorEager, name: str
) -> None:
    """A literal combined with a length-1 *column* does not become a scalar.

    Length-1 is only a proxy for scalar-like: a `unique()` which happens to return one
    row is a real column, so the result must not be broadcast back to the frame height.
    """
    pl_raises = POLARS_WHEN_RAISES if name == "when_then" else POLARS_EAGER_RAISES
    if "polars" in str(constructor_eager) and not pl_raises:
        reason = "Polars broadcasts the length-1 result, see https://github.com/pola-rs/polars/issues/26023"
        request.applymarker(pytest.mark.xfail(reason=reason))
    df = nw.from_native(constructor_eager({"a": [1, 1, 1]}))
    with pytest.raises(ShapeError):
        df.select(x=LENGTH_CHANGING[name], y=nw.col("a"))
