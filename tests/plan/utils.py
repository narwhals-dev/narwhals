from __future__ import annotations

import re
from typing import TYPE_CHECKING, Any

import pytest

import narwhals as nw
from narwhals import _plan as nwp
from narwhals._plan import Expr, Selector, _expansion, _parse, expressions as ir
from narwhals._utils import qualified_type_name
from tests.utils import assert_equal_data as _assert_equal_data

pytest.importorskip("pyarrow")

import pyarrow as pa

if TYPE_CHECKING:
    import sys
    from collections.abc import Iterable, Mapping, Sequence

    from typing_extensions import LiteralString, TypeAlias

    from narwhals._plan.typing import IntoExpr, OneOrIterable, Seq
    from narwhals.typing import IntoSchema

    if sys.version_info >= (3, 11):
        _Flags: TypeAlias = "int | re.RegexFlag"
    else:
        _Flags: TypeAlias = int


def first(*names: str | Sequence[str]) -> nwp.Expr:
    return nwp.col(*names).first()


def last(*names: str | Sequence[str]) -> nwp.Expr:
    return nwp.col(*names).last()


class Frame:
    """Schema-only `{Expr,Selector}` projection testing tool.

    Arguments:
        schema: A Narwhals Schema.

    Examples:
        >>> import narwhals as nw
        >>> import narwhals._plan.selectors as ncs
        >>> df = Frame.from_mapping(
        ...     {
        ...         "abc": nw.UInt16(),
        ...         "bbb": nw.UInt32(),
        ...         "cde": nw.Float64(),
        ...         "def": nw.Float32(),
        ...         "eee": nw.Boolean(),
        ...     }
        ... )

        Determine the columns names that expression input would select

        >>> df.project_names(ncs.numeric() - ncs.by_index(1, 2))
        ('abc', 'def')

        Assert an expression selects names in a given order

        >>> df.assert_selects(ncs.by_name("eee", "abc"), "eee", "abc")

        Raising a helpful error if something went wrong

        >>> df.assert_selects(ncs.duration(), "eee", "abc")
        Traceback (most recent call last):
        AssertionError: Projected column names do not match expected names:
        result  : ()
        expected: ('eee', 'abc')
    """

    def __init__(self, schema: nw.Schema) -> None:
        self.schema = schema
        self.columns = tuple(schema.names())

    @staticmethod
    def from_mapping(mapping: IntoSchema) -> Frame:
        """Construct from inputs accepted in `nw.Schema`."""
        return Frame(nw.Schema(mapping))

    @staticmethod
    def from_names(*column_names: str) -> Frame:
        """Construct with all `nw.Int64()`."""
        return Frame(nw.Schema((name, nw.Int64()) for name in column_names))

    @property
    def width(self) -> int:
        """Get the number of columns in the schema."""
        return len(self.columns)

    def project(
        self, exprs: OneOrIterable[IntoExpr], *more_exprs: IntoExpr
    ) -> Seq[ir.NamedIR]:
        """Parse and expand expressions into named representations.

        Arguments:
            exprs: Column(s) to select. Accepts expression input. Strings are parsed as column names,
                other non-expression inputs are parsed as literals.
            *more_exprs: Column(s) to select, specified as positional arguments.

        Note:
            `NamedIR` is the form of expression passed to the compliant-level.

        Examples:
            >>> import datetime as dt
            >>> import narwhals._plan.selectors as ncs
            >>> df = Frame.from_names("a", "b", "c", "d", "idx1", "idx2")
            >>> expr_1 = (
            ...     ncs.by_name("a", "d")
            ...     .first()
            ...     .over(ncs.by_index(range(1, 4)), order_by=ncs.matches(r"idx"))
            ... )
            >>> expr_2 = (ncs.by_name("a") | ncs.by_index(2)).abs().name.suffix("_abs")
            >>> expr_3 = dt.date(2000, 1, 1)

            >>> df.project(expr_1, expr_2, expr_3)  # doctest: +NORMALIZE_WHITESPACE
            (a=col('a').first().over(partition_by=[col('b'), col('c'), col('d')], order_by=[col('idx1'), col('idx2')]),
             d=col('d').first().over(partition_by=[col('b'), col('c'), col('d')], order_by=[col('idx1'), col('idx2')]),
             a_abs=col('a').abs(),
             c_abs=col('c').abs(),
             literal=lit(date: 2000-01-01))
        """
        expr_irs = _parse.parse_into_seq_of_expr_ir(exprs, *more_exprs)
        named_irs, _ = _expansion.prepare_projection(expr_irs, schema=self.schema)
        return named_irs

    def project_names(self, *exprs: IntoExpr) -> Seq[str]:
        named_irs = self.project(*exprs)
        return tuple(e.name for e in named_irs)

    def assert_selects(self, selector: Selector | Expr, *column_names: str) -> None:
        result = self.project_names(selector)
        expected = column_names
        assert result == expected, (
            f"Projected column names do not match expected names:\n"
            f"result  : {result!r}\n"
            f"expected: {expected!r}"
        )


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


def assert_not_selector(actual: Expr | Selector, /) -> None:
    """Assert that `actual` was converted into an `Expr`."""
    assert isinstance(actual, Expr), (
        f"Didn't expect you to pass a {qualified_type_name(actual)!r} here, got: {actual!r}"
    )
    assert not isinstance(actual, Selector), (
        f"This operation should have returned `Expr`, but got {qualified_type_name(actual)!r}\n{actual!r}"
    )


def is_expr_ir_equal(actual: Expr | ir.ExprIR, expected: Expr | ir.ExprIR, /) -> bool:
    """Return True if `actual` is equivalent to `expected`.

    Note:
        Prefer `assert_expr_ir_equal` unless you need a `bool` for branching.
    """
    return _unwrap_ir(actual) == _unwrap_ir(expected)


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


def re_compile(
    pattern: str, flags: _Flags = re.DOTALL | re.IGNORECASE
) -> re.Pattern[str]:
    """Compile a regular expression pattern, returning a Pattern object.

    Helper to default to using `flags=re.DOTALL | re.IGNORECASE`.
    """
    return re.compile(pattern, flags)
