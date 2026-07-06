from __future__ import annotations

from typing import TYPE_CHECKING, Any, final

from altair.utils import is_undefined

from narwhals._plan import expressions as ir

if TYPE_CHECKING:
    from collections.abc import Iterator

    import altair as alt

    from narwhals._plan.expr import Expr as NwExpr


@final
class _ParameterIR(ir.ExprIR):
    """A wrapper around an [`altair.Parameter`][]."""

    __slots__ = ("parameter",)
    parameter: alt.Parameter

    @property
    def name(self) -> str:
        return self.parameter.name

    def is_scalar(self) -> bool:
        return True

    def __repr__(self) -> str:
        return f"param(name={self.name!r})"

    @property
    def __immutable_values__(self) -> Iterator[Any]:
        yield (
            type(self.parameter),
            self.name,
            type(self.parameter.param),
            self.parameter.param_type,
        )

    @staticmethod
    def from_altair(parameter: alt.Parameter, /) -> _ParameterIR:
        # NOTE: `alt.Parameter` is constructed in a very unusual way & the typing doesn't reflect the final state.
        # I'll be checking this once and then ignoring everything post-init
        if (
            is_undefined(parameter.param)
            or is_undefined(parameter.param_type)
            or (not isinstance(parameter.name, str))
            or parameter.name == "__TEMP__"
        ):
            msg = f"Parameter {parameter.name!r} is incorrectly initialized, got: {parameter!r}"
            raise TypeError(msg)
        return _ParameterIR(parameter=parameter)


def from_altair(parameter: alt.Parameter, /) -> NwExpr:
    return _ParameterIR.from_altair(parameter).to_narwhals()


def to_altair(expr: NwExpr, /) -> alt.Parameter:
    """Try to return the wrapped [`altair.Parameter`][]."""
    e_ir = expr._ir
    if isinstance(e_ir, _ParameterIR):
        return e_ir.parameter
    msg = f"Expected a narwhals-wrapped altair parameter, got: {e_ir}"
    raise TypeError(msg)
