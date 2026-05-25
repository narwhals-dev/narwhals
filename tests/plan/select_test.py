from __future__ import annotations

import datetime as dt
from typing import TYPE_CHECKING, Any, Literal

import pytest

import narwhals as nw
import narwhals._plan as nwp
import narwhals._plan.selectors as ncs
from narwhals.exceptions import InvalidOperationError
from tests.plan.utils import Series, assert_equal_data, re_compile
from tests.utils import POLARS_VERSION, PYARROW_VERSION

if TYPE_CHECKING:
    import polars as pl
    import pyarrow as pa

    from narwhals._plan.typing import IntoExpr, OneOrIterable
    from narwhals._typing import Arrow, EagerAllowed, IntoBackend, LazyAllowed, Polars
    from tests.conftest import Data
    from tests.plan.utils import Eager, Lazy
if PYARROW_VERSION and PYARROW_VERSION < (21,):  # pragma: no cover
    pytest.importorskip("numpy")

cases = pytest.mark.parametrize(
    ("exprs", "named_exprs", "expected"),
    [
        ((None,), {}, {"literal": [None]}),
        (
            (1,),
            {"two": 2, "three": nwp.lit({"four": 5, "six": 7})},
            {"literal": [1], "two": [2], "three": [{"four": 5, "six": 7}]},
        ),
        ((), {"numbers": nwp.int_range(10)}, {"numbers": list(range(10))}),
        (
            (
                nwp.date_range(
                    dt.date(2020, 1, 1), dt.date(2021, 1, 1), dt.timedelta(weeks=17)
                ).alias("dates"),
                nwp.lit("123"),
            ),
            {},
            {
                "dates": [
                    dt.date(2020, 1, 1),
                    dt.date(2020, 4, 29),
                    dt.date(2020, 8, 26),
                    dt.date(2020, 12, 23),
                ],
                "literal": ["123"] * 4,
            },
        ),
        (
            (nwp.len().cast(nw.Boolean), nwp.lit(1).alias("lit")),
            {},
            {"len": [False], "lit": [1]},
        ),
    ],
    ids=["None", "mixed-positional-named", "int_range", "date_range", "len-0-cast"],
)


@cases
def test_lazy(
    exprs: tuple[OneOrIterable[IntoExpr], ...],
    named_exprs: dict[str, IntoExpr],
    expected: Data,
    lazy: Lazy,
) -> None:
    if (
        lazy == "polars"
        and POLARS_VERSION < (1, 17, 0)
        and expected == {"literal": [None]}
    ):
        # NOTE: This was a regression, but unclear when it stopped working
        pytest.skip(
            reason=(
                "PanicException: `arg_sort` operation not supported for dtype `null`.\n"
                "https://github.com/pola-rs/polars/pull/20135"
            )
        )
    result = nwp.select(*exprs, lazy=lazy, **named_exprs).collect().sort(ncs.first())
    assert_equal_data(result, expected)


@cases
def test_eager(
    exprs: tuple[OneOrIterable[IntoExpr], ...],
    named_exprs: dict[str, IntoExpr],
    expected: Data,
    eager: Eager,
) -> None:
    result = nwp.select(*exprs, eager=eager, **named_exprs)
    assert_equal_data(result, expected)


def test_eager_series(series: Series) -> None:
    ser = series([1, 2, 3], name="a")
    result = nwp.select(ser, b=nwp.lit("four"), eager=series.implementation)
    expected = {"a": [1, 2, 3], "b": ["four", "four", "four"]}
    assert_equal_data(result, expected)


def test_lazy_series(lazy: Lazy) -> None:
    # NOTE: Will work for any backend that supports `LitSeries`
    lf = nwp.select([1, 5, 7, 9], lazy=lazy).explode(ncs.first())
    ser = lf.collect().to_series()
    result = nwp.select(a=1, b=2, c=ser, lazy=lazy).sort(ncs.last()).collect()
    expected = {"a": [1] * 4, "b": [2] * 4, "c": [1, 5, 7, 9]}
    assert_equal_data(result, expected)


def test_empty_lazy(lazy: Lazy) -> None:
    with pytest.raises(TypeError, match=re_compile(r"at least one.+expression")):
        nwp.select(lazy=lazy)


def test_empty_eager(eager: Eager) -> None:
    result = nwp.select(eager=eager)
    assert result.shape == (0, 0)
    assert_equal_data(result, {})


def test_neither() -> None:
    pattern = re_compile(r"either.+may be None")
    with pytest.raises(TypeError, match=pattern):
        nwp.select()  # type: ignore[call-overload]
    with pytest.raises(TypeError, match=pattern):
        nwp.select(eager=None)  # type: ignore[call-overload]
    with pytest.raises(TypeError, match=pattern):
        nwp.select(lazy=None)  # type: ignore[call-overload]
    with pytest.raises(TypeError, match=pattern):
        nwp.select(eager=None, lazy=None)  # type: ignore[call-overload]


def test_both() -> None:
    with pytest.raises(TypeError, match=re_compile(r"either.+may be provided")):
        nwp.select(1, eager="polars", lazy="polars")  # type: ignore[call-overload]


@pytest.mark.parametrize(
    ("exprs", "expected"),
    [
        (nwp.lit({"a": 1}).struct.field("a").alias("b"), {"b": [1]}),
        (
            [
                nwp.lit(0).alias("a"),
                nwp.lit({"a": 1, "b": 2}).struct.field("b").alias("c"),
            ],
            {"a": [0], "c": [2]},
        ),
    ],
    ids=["alone", "with-a-friend"],
)
def test_lit_struct_field_lazy(
    exprs: OneOrIterable[IntoExpr], expected: Data, lazy: Lazy
) -> None:
    result = nwp.select(exprs, lazy=lazy).collect()
    assert_equal_data(result, expected)  # pragma: no cover


@pytest.mark.parametrize(
    ("expr", "name"),
    [
        (nwp.lit({"a": 1}).struct.field("b"), "b"),
        (
            nwp.lit({"a": {"b": {"c": "d"}}})
            .struct.field("a")
            .struct.field("b")
            .struct.field("c")
            .struct.field("e"),
            "e",
        ),
    ],
    ids=["single", "nested"],
)
def test_lit_struct_field_missing_lazy(expr: nwp.Expr, name: str, lazy: Lazy) -> None:
    with pytest.raises(InvalidOperationError, match=f"Struct field not found {name!r}"):
        nwp.select(expr, lazy=lazy).collect()


if TYPE_CHECKING:
    from typing_extensions import assert_type

    def typing_exhaust_overloads(  # noqa: PLR0914, PLR0915
        polars: Literal["polars"],
        pyarrow: Literal["pyarrow"],
        either: Polars | Arrow,
        eager: IntoBackend[EagerAllowed],
        lazy: IntoBackend[LazyAllowed],
    ) -> None:
        a, b = nwp.lit(1).alias("a"), nwp.lit(2).alias("b")

        df_pl_1 = nwp.select(a, b, eager=polars)
        df_pl_2 = nwp.select(named=1, eager=polars)
        df_pl_3 = nwp.select(a, b, named=1, eager=polars)
        df_pl_4 = nwp.select(a, b, eager=polars, lazy=None)
        df_pl_5 = nwp.select(named=1, eager=polars, lazy=None)
        df_pl_6 = nwp.select(a, b, named=1, eager=polars, lazy=None)

        assert_type(df_pl_1, nwp.DataFrame[pl.DataFrame, pl.Series])
        assert_type(df_pl_2, nwp.DataFrame[pl.DataFrame, pl.Series])
        assert_type(df_pl_3, nwp.DataFrame[pl.DataFrame, pl.Series])
        assert_type(df_pl_4, nwp.DataFrame[pl.DataFrame, pl.Series])
        assert_type(df_pl_5, nwp.DataFrame[pl.DataFrame, pl.Series])
        assert_type(df_pl_6, nwp.DataFrame[pl.DataFrame, pl.Series])

        df_pa_1 = nwp.select(a, b, eager=pyarrow)
        df_pa_2 = nwp.select(named=1, eager=pyarrow)
        df_pa_3 = nwp.select(a, b, named=1, eager=pyarrow)
        df_pa_4 = nwp.select(a, b, eager=pyarrow, lazy=None)
        df_pa_5 = nwp.select(named=1, eager=pyarrow, lazy=None)
        df_pa_6 = nwp.select(a, b, named=1, eager=pyarrow, lazy=None)

        assert_type(df_pa_1, nwp.DataFrame[pa.Table, pa.ChunkedArray[Any]])
        assert_type(df_pa_2, nwp.DataFrame[pa.Table, pa.ChunkedArray[Any]])
        assert_type(df_pa_3, nwp.DataFrame[pa.Table, pa.ChunkedArray[Any]])
        assert_type(df_pa_4, nwp.DataFrame[pa.Table, pa.ChunkedArray[Any]])
        assert_type(df_pa_5, nwp.DataFrame[pa.Table, pa.ChunkedArray[Any]])
        assert_type(df_pa_6, nwp.DataFrame[pa.Table, pa.ChunkedArray[Any]])

        df_either_1 = nwp.select(a, b, eager=either)  # noqa: F841
        df_either_2 = nwp.select(named=1, eager=either)  # noqa: F841
        df_either_3 = nwp.select(a, b, named=1, eager=either)  # noqa: F841
        df_either_4 = nwp.select(a, b, eager=either, lazy=None)  # noqa: F841
        df_either_5 = nwp.select(named=1, eager=either, lazy=None)  # noqa: F841
        df_either_6 = nwp.select(a, b, named=1, eager=either, lazy=None)  # noqa: F841

        # TODO @dangotbanned: Decide on what to do for `either`
        # Probably don't want to *enforce* this to be `Any`, but it's okay if it is

        df_eager_1 = nwp.select(a, b, eager=eager)
        df_eager_2 = nwp.select(named=1, eager=eager)
        df_eager_3 = nwp.select(a, b, named=1, eager=eager)
        df_eager_4 = nwp.select(a, b, eager=eager, lazy=None)
        df_eager_5 = nwp.select(named=1, eager=eager, lazy=None)
        df_eager_6 = nwp.select(a, b, named=1, eager=eager, lazy=None)

        assert_type(df_eager_1, nwp.DataFrame[Any, Any])
        assert_type(df_eager_2, nwp.DataFrame[Any, Any])
        assert_type(df_eager_3, nwp.DataFrame[Any, Any])
        assert_type(df_eager_4, nwp.DataFrame[Any, Any])
        assert_type(df_eager_5, nwp.DataFrame[Any, Any])
        assert_type(df_eager_6, nwp.DataFrame[Any, Any])

        lf_pl_1 = nwp.select(a, b, lazy=polars)
        lf_pl_2 = nwp.select(named=1, lazy=polars)
        lf_pl_3 = nwp.select(a, b, named=1, lazy=polars)
        lf_pl_4 = nwp.select(a, b, lazy=polars, eager=None)
        lf_pl_5 = nwp.select(named=1, lazy=polars, eager=None)
        lf_pl_6 = nwp.select(a, b, named=1, lazy=polars, eager=None)

        assert_type(lf_pl_1, nwp.LazyFrame[pl.LazyFrame])
        assert_type(lf_pl_2, nwp.LazyFrame[pl.LazyFrame])
        assert_type(lf_pl_3, nwp.LazyFrame[pl.LazyFrame])
        assert_type(lf_pl_4, nwp.LazyFrame[pl.LazyFrame])
        assert_type(lf_pl_5, nwp.LazyFrame[pl.LazyFrame])
        assert_type(lf_pl_6, nwp.LazyFrame[pl.LazyFrame])

        lf_lazy_1 = nwp.select(a, b, lazy=lazy)
        lf_lazy_2 = nwp.select(named=1, lazy=lazy)
        lf_lazy_3 = nwp.select(a, b, named=1, lazy=lazy)
        lf_lazy_4 = nwp.select(a, b, lazy=lazy, eager=None)
        lf_lazy_5 = nwp.select(named=1, lazy=lazy, eager=None)
        lf_lazy_6 = nwp.select(a, b, named=1, lazy=lazy, eager=None)

        assert_type(lf_lazy_1, nwp.LazyFrame[Any])
        assert_type(lf_lazy_2, nwp.LazyFrame[Any])
        assert_type(lf_lazy_3, nwp.LazyFrame[Any])
        assert_type(lf_lazy_4, nwp.LazyFrame[Any])
        assert_type(lf_lazy_5, nwp.LazyFrame[Any])
        assert_type(lf_lazy_6, nwp.LazyFrame[Any])
