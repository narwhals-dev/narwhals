from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pytest

from narwhals import _plan as nwp
from narwhals._plan import expressions as ir, selectors as ncs
from tests.utils import assert_equal_data as _assert_equal_data

pytest.importorskip("pyarrow")

import pyarrow as pa

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping, Sequence

    from typing_extensions import LiteralString


def cols(*names: str | Sequence[str]) -> nwp.Expr:
    return ncs.by_name(*names).as_expr()


def nth(*indices: int | Sequence[int]) -> nwp.Expr:
    return ncs.by_index(*indices).as_expr()


def first(*names: str | Sequence[str]) -> nwp.Expr:
    return cols(*names).first()


def last(*names: str | Sequence[str]) -> nwp.Expr:
    return cols(*names).last()


def _unwrap_ir(obj: nwp.Expr | ir.ExprIR | ir.NamedIR) -> ir.ExprIR:
    if isinstance(obj, nwp.Expr):
        return obj._ir
    if isinstance(obj, ir.ExprIR):
        return obj
    if isinstance(obj, ir.NamedIR):
        return obj.expr
    raise NotImplementedError(type(obj))


def assert_expr_ir_equal(
    actual: nwp.Expr | ir.ExprIR | ir.NamedIR,
    expected: nwp.Expr | ir.ExprIR | ir.NamedIR | LiteralString,
    /,
) -> None:
    """Assert that `actual` is equivalent to `expected`.

    Arguments:
        actual: Result expression or IR to compare.
        expected: Target expression, IR, or repr to compare.

    Notes:
        Performing a repr comparison is more fragile, so should be avoided
        *unless* we raise an error at creation time.
    """
    lhs = _unwrap_ir(actual)
    if isinstance(expected, str):
        assert repr(lhs) == expected, (
            f"\nlhs:\n    {lhs!r}\n\nexpected:\n    {expected!r}"
        )
    elif isinstance(actual, ir.NamedIR) and isinstance(expected, ir.NamedIR):
        assert actual == expected, (
            f"\nactual:\n    {actual!r}\n\nexpected:\n    {expected!r}"
        )
    else:
        rhs = expected._ir if isinstance(expected, nwp.Expr) else expected
        assert lhs == rhs, f"\nlhs:\n    {lhs!r}\n\nrhs:\n    {rhs!r}"


def named_ir(name: str, expr: nwp.Expr | ir.ExprIR, /) -> ir.NamedIR[ir.ExprIR]:
    """Helper constructor for test compare."""
    return ir.NamedIR(expr=expr._ir if isinstance(expr, nwp.Expr) else expr, name=name)


def dataframe(data: dict[str, Any], /) -> nwp.DataFrame[pa.Table, pa.ChunkedArray[Any]]:
    return nwp.DataFrame.from_native(pa.table(data))


def series(values: Iterable[Any], /) -> nwp.Series[pa.ChunkedArray[Any]]:
    return nwp.Series.from_native(pa.chunked_array([values]))


def assert_equal_data(
    result: nwp.DataFrame[Any, Any], expected: Mapping[str, Any]
) -> None:
    _assert_equal_data(result.to_dict(as_series=False), expected)
