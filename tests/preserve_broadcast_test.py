from __future__ import annotations

import pytest

import narwhals as nw
from narwhals.exceptions import ShapeError
from tests.utils import POLARS_VERSION, ConstructorEager, assert_equal_data

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
