from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

import narwhals._plan as nwp
from narwhals.exceptions import InvalidOperationError
from tests.plan.utils import assert_equal_data, re_compile

if TYPE_CHECKING:
    from narwhals._plan.typing import IntoExpr, OneOrIterable
    from tests.conftest import Data
    from tests.plan.utils import Eager, Lazy


XFAIL_TODO = pytest.mark.xfail(reason="TODO")


@XFAIL_TODO
def test_lazy(
    exprs: tuple[OneOrIterable[IntoExpr], ...],
    named_exprs: dict[str, IntoExpr],
    expected: Data,
    lazy: Lazy,
) -> None:
    result = nwp.select(*exprs, lazy=lazy, **named_exprs)
    assert_equal_data(result.collect(), expected)


@XFAIL_TODO
@pytest.mark.parametrize(
    ("exprs", "named_exprs", "expected"),
    [((), {"numbers": nwp.int_range(10)}, {"numbers": list(range(10))})],
)
def test_eager(
    exprs: tuple[OneOrIterable[IntoExpr], ...],
    named_exprs: dict[str, IntoExpr],
    expected: Data,
    eager: Eager,
) -> None:
    result = nwp.select(*exprs, eager=eager, **named_exprs)
    assert_equal_data(result, expected)


@XFAIL_TODO
def test_empty_lazy(lazy: Lazy) -> None:
    with pytest.raises(InvalidOperationError, match=re_compile(r"at least one.+expr")):
        nwp.select(lazy=lazy)


@XFAIL_TODO
def test_empty_eager(eager: Eager) -> None:
    result = nwp.select(eager=eager)
    assert result.shape == (0, 0)
    assert_equal_data(result, {})


@XFAIL_TODO
def test_neither() -> None:
    pattern = re_compile(r"either.+may be None")
    with pytest.raises(TypeError, match=pattern):
        nwp.select()
    with pytest.raises(TypeError, match=pattern):
        nwp.select(eager=None)
    with pytest.raises(TypeError, match=pattern):
        nwp.select(lazy=None)
    with pytest.raises(TypeError, match=pattern):
        nwp.select(eager=None, lazy=None)


@XFAIL_TODO
def test_both() -> None:
    with pytest.raises(TypeError, match=re_compile(r"either.+may be provided")):
        nwp.select(1, eager="polars", lazy="polars")


if TYPE_CHECKING:

    def typing_exhaust_overloads() -> None:
        df_pl = nwp.select(1, 2, 3, eager="polars")
        df_pa = nwp.select(1, 2, 3, eager="pyarrow")
        lf_pl = nwp.select(1, 2, 3, lazy="polars")
        # TODO @dangotbanned: Asserts
        # TODO @dangotbanned: More combinations of `*args`/`**kwds`
