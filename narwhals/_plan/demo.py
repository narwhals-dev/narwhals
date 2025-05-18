from __future__ import annotations

import builtins
import typing as t

from narwhals._plan import boolean
from narwhals._plan import functions as F  # noqa: N812
from narwhals._plan.dummy import DummySeries
from narwhals._plan.expr import All
from narwhals._plan.expr import Column
from narwhals._plan.expr import Columns
from narwhals._plan.expr import IndexColumns
from narwhals._plan.expr import Len
from narwhals._plan.expr import Nth
from narwhals._plan.literal import ScalarLiteral
from narwhals._plan.literal import SeriesLiteral
from narwhals._plan.strings import ConcatHorizontal
from narwhals.dtypes import DType
from narwhals.dtypes import Unknown
from narwhals.utils import flatten

if t.TYPE_CHECKING:
    from narwhals._plan.dummy import DummyExpr
    from narwhals.typing import NonNestedLiteral


def col(*names: str | t.Iterable[str]) -> DummyExpr:
    flat_names = tuple(flatten(names))
    node = (
        Column(name=flat_names[0])
        if builtins.len(flat_names) == 1
        else Columns(names=flat_names)
    )
    return node.to_narwhals()


def nth(*indices: int | t.Sequence[int]) -> DummyExpr:
    flat_indices = tuple(flatten(indices))
    node = (
        Nth(index=flat_indices[0])
        if builtins.len(flat_indices) == 1
        else IndexColumns(indices=flat_indices)
    )
    return node.to_narwhals()


def lit(
    value: NonNestedLiteral | DummySeries, dtype: DType | type[DType] | None = None
) -> DummyExpr:
    if isinstance(value, DummySeries):
        return SeriesLiteral(value=value).to_literal().to_narwhals()
    if dtype is None or not isinstance(dtype, DType):
        dtype = Unknown()
    return ScalarLiteral(value=value, dtype=dtype).to_literal().to_narwhals()


def len() -> DummyExpr:
    return Len().to_narwhals()


def all() -> DummyExpr:
    return All().to_narwhals()


def max(*columns: str) -> DummyExpr:
    return col(columns).max()


def mean(*columns: str) -> DummyExpr:
    return col(columns).mean()


def min(*columns: str) -> DummyExpr:
    return col(columns).min()


def median(*columns: str) -> DummyExpr:
    return col(columns).median()


def sum(*columns: str) -> DummyExpr:
    return col(columns).sum()


def all_horizontal(*exprs: DummyExpr | t.Iterable[DummyExpr]) -> DummyExpr:
    it = (expr._ir for expr in flatten(exprs))
    return boolean.AllHorizontal().to_function_expr(*it).to_narwhals()


def any_horizontal(*exprs: DummyExpr | t.Iterable[DummyExpr]) -> DummyExpr:
    it = (expr._ir for expr in flatten(exprs))
    return boolean.AnyHorizontal().to_function_expr(*it).to_narwhals()


def sum_horizontal(*exprs: DummyExpr | t.Iterable[DummyExpr]) -> DummyExpr:
    it = (expr._ir for expr in flatten(exprs))
    return F.SumHorizontal().to_function_expr(*it).to_narwhals()


def min_horizontal(*exprs: DummyExpr | t.Iterable[DummyExpr]) -> DummyExpr:
    it = (expr._ir for expr in flatten(exprs))
    return F.MinHorizontal().to_function_expr(*it).to_narwhals()


def max_horizontal(*exprs: DummyExpr | t.Iterable[DummyExpr]) -> DummyExpr:
    it = (expr._ir for expr in flatten(exprs))
    return F.MaxHorizontal().to_function_expr(*it).to_narwhals()


def mean_horizontal(*exprs: DummyExpr | t.Iterable[DummyExpr]) -> DummyExpr:
    it = (expr._ir for expr in flatten(exprs))
    return F.MeanHorizontal().to_function_expr(*it).to_narwhals()


def concat_str(
    exprs: DummyExpr | t.Iterable[DummyExpr],
    *more_exprs: DummyExpr,
    separator: str = "",
    ignore_nulls: bool = False,
) -> DummyExpr:
    it = (expr._ir for expr in flatten([*flatten([exprs]), *more_exprs]))
    return (
        ConcatHorizontal(separator=separator, ignore_nulls=ignore_nulls)
        .to_function_expr(*it)
        .to_narwhals()
    )
