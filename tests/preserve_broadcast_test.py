from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING, Any, cast

import pytest

import narwhals as nw
from narwhals.exceptions import ShapeError
from tests.utils import (
    PANDAS_VERSION,
    POLARS_VERSION,
    ConstructorEager,
    assert_equal_data,
)

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
    NestedCase: TypeAlias = "tuple[list[Any], Op]"


def like(series: EagerSeriesAny, data: Data) -> EagerSeriesAny:
    """An operand standing for the same thing as `series`: scalar-like iff `series` is."""
    result = series.from_iterable(data * len(series), context=series)
    result._broadcast = series._broadcast and len(result) == 1
    return result


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
    "dt.truncate": ([datetime(2021, 1, 2)], lambda s: s.dt.truncate("1d")),
    "dt.year": ([datetime(2021, 1, 2)], lambda s: s.dt.year()),
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
    "str.len_chars": (["abc"], lambda s: s.str.len_chars()),
    "str.slice": (["abc"], lambda s: s.str.slice(1, 2)),
    "str.strip_chars": ([" a "], lambda s: s.str.strip_chars(None)),
    "str.to_datetime": (["2021-01-02"], lambda s: s.str.to_datetime("%Y-%m-%d")),
    "str.to_uppercase": (["a"], lambda s: s.str.to_uppercase()),
    "zip_with": (["a"], lambda s: s.zip_with(like(s, [True]), like(s, ["b"]))),
}

LOSES_LENGTH_1: dict[str, Case] = {
    # NOTE: a real boolean column, not a scalar-like mask; `Expr.filter(nw.lit(...))`
    # is broken on both pandas and pyarrow, independently of `_broadcast`.
    "filter": ([1], lambda s: s.filter(s.from_iterable([False], context=s))),
    "drop_nulls": ([None], lambda s: s.drop_nulls()),
    "head": ([1], lambda s: s.head(0)),
    "_align_full_broadcast": (
        [1],
        lambda s: s._align_full_broadcast(s, like(s, [1, 2, 3]))[0],
    ),
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

LIST_INT = nw.List(nw.Int64())
STRUCT_INT = nw.Struct({"x": nw.Int64()})

# Nested dtypes need `pyarrow` (and `pandas>=2.2`), so they get their own fixture.
NESTED: dict[str, NestedCase] = {
    "list.get": ([[1, 2]], lambda s: s.cast(LIST_INT).list.get(0)),
    "list.len": ([[1, 2]], lambda s: s.cast(LIST_INT).list.len()),
    "struct.field": ([{"x": 1}], lambda s: s.cast(STRUCT_INT).struct.field("x")),
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
        (nw.col("a") + nw.lit(1), [2]),
        (nw.col("a").sum() + nw.col("a"), [2]),
        (nw.col("a") + nw.col("a").sum(), [2]),
        (nw.lit(5).clip(nw.col("a"), nw.col("a") + 10), [5]),
        (nw.col("a").clip(nw.lit(5), nw.lit(10)), [5]),
        (nw.when(nw.col("a") > 0).then(nw.lit(7)).otherwise(nw.col("a")), [7]),
        (nw.sum_horizontal(nw.lit(1), nw.col("a")), [2]),
        (nw.concat_str(nw.lit("!"), nw.col("s")), ["!x"]),
        (nw.concat_str(nw.col("s"), nw.lit("!")), ["x!"]),
    ],
)
def test_scalar_operand_single_row(
    constructor_eager: ConstructorEager, expr: nw.Expr, expected: list[int | str]
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


@pytest.fixture
def nested_compliant_series(
    compliant_series: CompliantSeries, constructor_eager: ConstructorEager
) -> CompliantSeries:
    pytest.importorskip("pyarrow")
    if any(x in str(constructor_eager) for x in ("modin", "cudf")):
        pytest.skip("nested dtypes are not supported")
    if "pandas" in str(constructor_eager) and PANDAS_VERSION < (2, 2):
        pytest.skip("nested dtypes require pandas>=2.2")
    return compliant_series


@pytest.mark.parametrize(("data", "op"), NESTED.values(), ids=NESTED)
def test_nested_namespace_broadcast(
    nested_compliant_series: CompliantSeries, data: list[Any], op: Op
) -> None:
    series = nested_compliant_series(data)
    series._broadcast = True
    assert op(series)._broadcast is True
    assert op(nested_compliant_series(data * 3))._broadcast is False
