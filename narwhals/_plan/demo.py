from __future__ import annotations

import builtins
import typing as t

from narwhals._plan.common import DummySeries
from narwhals._plan.expr import All
from narwhals._plan.expr import Column
from narwhals._plan.expr import Columns
from narwhals._plan.expr import FunctionExpr
from narwhals._plan.expr import IndexColumns
from narwhals._plan.expr import Len
from narwhals._plan.expr import Nth
from narwhals._plan.functions import SumHorizontal
from narwhals._plan.literal import ScalarLiteral
from narwhals._plan.literal import SeriesLiteral
from narwhals._plan.options import FunctionOptions

if t.TYPE_CHECKING:
    from narwhals._plan.common import DummyExpr
    from narwhals.dtypes import DType
    from narwhals.typing import NonNestedLiteral


def col(*names: str | t.Iterable[str]) -> DummyExpr:
    from narwhals.utils import flatten

    flat_names = tuple(flatten(names))
    node = (
        Column(name=flat_names[0])
        if builtins.len(flat_names) == 1
        else Columns(names=flat_names)
    )
    return node.to_narwhals()


def nth(*indices: int | t.Sequence[int]) -> DummyExpr:
    from narwhals.utils import flatten

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
    from narwhals.dtypes import DType
    from narwhals.dtypes import Unknown

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


def sum_horizontal(*exprs: DummyExpr | t.Iterable[DummyExpr]) -> DummyExpr:
    from narwhals.utils import flatten

    flat_exprs = tuple(flatten(exprs))
    # NOTE: Still need to figure out how these should be generated
    # Feel like it should be the union of `input` & `function`
    PLACEHOLDER = FunctionOptions.default()  # noqa: N806
    return FunctionExpr(
        input=flat_exprs, function=SumHorizontal(), options=PLACEHOLDER
    ).to_narwhals()
