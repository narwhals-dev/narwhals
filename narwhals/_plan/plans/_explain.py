"""Visual representation of query plans."""

from __future__ import annotations

from functools import singledispatch
from typing import TYPE_CHECKING, Any, Final

from narwhals._plan.plans import plan as lp

if TYPE_CHECKING:
    from collections.abc import Iterator, Mapping

    from typing_extensions import LiteralString

    from narwhals._plan.typing import Seq


_NO_ATTRS: Final[Mapping[type[lp.LogicalPlan], LiteralString]] = {
    lp.Collect: "SINK (memory)",
    lp.SinkParquet: "SINK (file)",
    lp.Pivot: "PIVOT[...]",  # NOTE: Only exists in `DslPlan`, not `IR` which defines the displays
    lp.HConcat: "HCONCAT",  # had extra indent
    lp.VConcat: "UNION",  # had extra indent
}

_INDENT_INCREMENT = 2
_SPACE = " "


# TODO @dangotbanned: Add a small cache
def pad(indent_level: int, /) -> LiteralString:
    return _SPACE * indent_level


# TODO @dangotbanned: Add a small cache
def next_level(indent_level: int, /) -> int:
    return indent_level + _INDENT_INCREMENT


def _seq(obj: Seq[Any], /) -> str:
    """Format a tuple using `list.__repr__`."""
    return repr(list(obj))


def _format(plan: lp.LogicalPlan, indent: int) -> str:
    it = _iter_format(plan, indent)
    if indent:
        pad_ = pad(indent)
        it = (f"{pad_}{line}" for line in it)
    return "\n".join(it)


def _format_recursive(plan: lp.LogicalPlan, indent: int) -> str:
    # `IRDisplay._format`
    # (here) https://github.com/pola-rs/polars/blob/40c171f9725279cd56888f443bd091eea79e5310/crates/polars-plan/src/plans/ir/format.rs#L259-L265
    # (overrides) https://github.com/pola-rs/polars/blob/40c171f9725279cd56888f443bd091eea79e5310/crates/polars-plan/src/plans/ir/format.rs#L148-L229
    result = "\n".join(_iter_format_recursive(plan, indent))
    return result if not indent else "\n" + result


@singledispatch
def _iter_format(plan: lp.LogicalPlan, indent: int) -> Iterator[str]:
    # `ir::format::write_ir_non_recursive`
    # https://github.com/pola-rs/polars/blob/40c171f9725279cd56888f443bd091eea79e5310/crates/polars-plan/src/plans/ir/format.rs#L705-L1006
    raise NotImplementedError(type(plan))


@singledispatch
def _iter_format_recursive(plan: lp.LogicalPlan, indent: int) -> Iterator[str]:
    if not plan.has_inputs:
        yield _format(plan, indent)
        return

    sub_indent = next_level(indent)
    it = _iter_format(plan, indent)
    if indent:
        pad_ = pad(indent)
        it = (f"{pad_}{line}" for line in it)
    yield from it
    for node in plan.iter_inputs():
        yield from _iter_format_recursive(node, sub_indent)


def explain(plan: lp.LogicalPlan) -> str:
    """Create a string representation of the query plan."""
    return _format_recursive(plan, 0)


@_iter_format.register(lp.Collect)
@_iter_format.register(lp.SinkParquet)
@_iter_format.register(lp.Pivot)
@_iter_format.register(lp.HConcat)
@_iter_format.register(lp.VConcat)
def _(plan: lp.LogicalPlan, *_: Any) -> Iterator[str]:
    yield _NO_ATTRS[type(plan)]


@_iter_format.register(lp.ScanDataFrame)
def _(plan: lp.ScanDataFrame, *_: Any) -> Iterator[str]:
    names = plan.schema.names
    fmt_names = ""
    if n_columns := len(names):
        fmt_names = ", ".join(f'"{name}"' for name in names[:4])
        if n_columns > 4:
            fmt_names += ", ..."
    yield f"DF [{fmt_names}]; {n_columns} COLUMNS"


@_iter_format.register(lp.ScanFile)
def _(plan: lp.ScanFile, *_: Any) -> Iterator[str]:
    yield f"{type(plan).__name__.removeprefix('Scan')} SCAN [{plan.source}]"


@_iter_format.register(lp.Select)
def _(plan: lp.Select, *_: Any) -> Iterator[str]:
    yield f"SELECT {_seq(plan.exprs)}"


@_iter_format.register(lp.SelectNames)
def _(plan: lp.SelectNames, *_: Any) -> Iterator[str]:
    yield f"SELECT {_seq(plan.names)}"


@_iter_format.register(lp.WithColumns)
def _(plan: lp.WithColumns, *_: Any) -> Iterator[str]:
    # has extra (uneven) indents upstream
    yield from (" WITH_COLUMNS:", f" {_seq(plan.exprs)}")


@_iter_format.register(lp.Filter)
def _(plan: lp.Filter, *_: Any) -> Iterator[str]:
    yield from (f"FILTER {plan.predicate!r}", "FROM")


@_iter_format.register(lp.Unique)
def _(plan: lp.Unique, *_: Any) -> Iterator[str]:
    opts = plan.options
    maintain = "maintain_order: True, " if opts.maintain_order else ""
    subset = f"subset: {_seq(s)}" if (s := plan.subset) else ""
    yield f"UNIQUE[{', '.join((maintain, f'keep: {opts.keep}', subset))}]"
    if isinstance(plan, lp.UniqueBy):
        yield f" BY {_seq(plan.order_by)}"


@_iter_format.register(lp.Sort)
def _(plan: lp.Sort, *_: Any) -> Iterator[str]:
    opts = plan.options
    s = f"SORT BY[{', '.join(repr(e) for e in plan.by)}"
    if any(opts.descending):
        s += f", descending: {_seq(opts.descending)}"
    if any(opts.nulls_last):
        s += f", nulls_last: {_seq(opts.nulls_last)}"
    yield f"{s}]"


@_iter_format.register(lp.Slice)
def _(plan: lp.Slice, *_: Any) -> Iterator[str]:
    yield f"SLICE[offset: {plan.offset}, len: {plan.length}]"


@_iter_format.register(lp.MapFunction)
def _(plan: lp.MapFunction, *_: Any) -> Iterator[str]:
    yield repr(plan.function)


# TODO @dangotbanned: Figure out why first line needed extra indent when nested
@_iter_format.register(lp.GroupBy)
def _(plan: lp.GroupBy, indent: int) -> Iterator[str]:
    sub_pad = pad(next_level(indent))
    yield from ("AGGREGATE", f"{sub_pad}{_seq(plan.aggs)} BY {_seq(plan.keys)}")


@_iter_format.register(lp.JoinAsof)
@_iter_format.register(lp.Join)
def _(plan: lp.Join | lp.JoinAsof, *_: Any) -> Iterator[str]:
    if isinstance(plan, lp.Join):
        how = plan.options.how.upper()
        if how == "CROSS":
            yield f"{how} JOIN"
            return
        left_on, right_on = plan.left_on, plan.right_on
    else:
        how = "ASOF"
        left_on, right_on = (plan.left_on,), (plan.right_on,)
    yield from (
        f"{how} JOIN:",
        f"LEFT PLAN ON: {_seq(left_on)}",
        f"RIGHT PLAN ON: {_seq(right_on)}",
    )


@_iter_format_recursive.register(lp.JoinAsof)
@_iter_format_recursive.register(lp.Join)
def _(plan: lp.Join | lp.JoinAsof, indent: int) -> Iterator[str]:
    sub_indent = next_level(indent)
    if isinstance(plan, lp.Join):
        how = plan.options.how.upper()
        left_on, right_on = plan.left_on, plan.right_on
    else:
        how = "ASOF"
        left_on, right_on = (plan.left_on,), (plan.right_on,)
    yield f"{how} JOIN:"
    left = _iter_format_recursive(plan.inputs[0], sub_indent)
    right = _iter_format_recursive(plan.inputs[1], sub_indent)
    if how == "CROSS":
        l_on, r_on = "LEFT PLAN:", "RIGHT PLAN:"
    else:
        l_on, r_on = (
            f"LEFT PLAN ON: {_seq(left_on)}",
            f"RIGHT PLAN ON: {_seq(right_on)}",
        )
    yield l_on
    yield from left
    yield r_on
    yield from right
    yield f"END {how} JOIN"


# NOTE: Watch out for weird indent issues
@_iter_format_recursive.register(lp.GroupBy)
def _(plan: lp.GroupBy, indent: int) -> Iterator[str]:
    sub_indent = next_level(indent)
    yield from _iter_format(plan, indent)
    yield f"{pad(sub_indent)}FROM"
    yield from _iter_format_recursive(plan.input, sub_indent)


@_iter_format_recursive.register(lp.VConcat)
@_iter_format_recursive.register(lp.HConcat)
def _(plan: lp.VConcat | lp.HConcat, indent: int) -> Iterator[str]:
    # Only includes sub indents
    sub_indent = next_level(indent)
    sub_sub_indent = next_level(sub_indent)
    sub_pad = pad(sub_indent)
    yield from _iter_format(plan, indent)
    for idx, input in enumerate(plan.inputs):
        yield f"{sub_pad}PLAN {idx}:"
        yield from _iter_format_recursive(input, sub_sub_indent)
    yield f"END {_NO_ATTRS[type(plan)]}"  # had extra indent
