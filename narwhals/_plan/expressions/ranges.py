from __future__ import annotations

import datetime as dt
from typing import TYPE_CHECKING, Any, ClassVar, Generic

import narwhals._plan.dtypes_mapper as dtm
from narwhals._plan import _parameters as params
from narwhals._plan._dispatch import DispatcherOptions
from narwhals._plan._dtype import ResolveDType
from narwhals._plan._function import BinaryFunction
from narwhals._plan.typing import NonNestedLiteralT_co as T_co
from narwhals._utils import qualified_type_name
from narwhals.exceptions import InvalidOperationError

if TYPE_CHECKING:
    from typing_extensions import Self

    from narwhals._plan.expressions import RangeExpr
    from narwhals.dtypes import IntegerType
    from narwhals.typing import ClosedInterval

# NOTE: See https://github.com/astral-sh/ty/issues/1777#issuecomment-3618906859
get_dtype = ResolveDType.get_dtype
map_all = ResolveDType.function.map_all
namespaced = DispatcherOptions.namespaced


class RangeFunction(BinaryFunction, Generic[T_co], dispatch=namespaced()):
    __function_parameters__: ClassVar = params.Binary(params.SCALAR, params.SCALAR)

    @classmethod
    def __function_expr__(cls) -> type[RangeExpr[Any]]:
        from narwhals._plan.expressions import RangeExpr

        return RangeExpr

    @classmethod
    def _valid_types(cls) -> tuple[type[T_co], ...]:
        raise NotImplementedError

    @classmethod
    def ensure_py_scalars(
        cls, start: Any, end: Any, eager: Any | None = None
    ) -> tuple[T_co, T_co]:
        valid_types = cls._valid_types()
        if isinstance(start, valid_types) and isinstance(end, valid_types):
            return start, end
        tp_names = " | ".join(tp.__name__ for tp in valid_types)
        name = cls.__expr_ir_dispatch__.name
        msg = f"All inputs for `{name}()` must resolve to {tp_names}, but got: ({qualified_type_name(start)}, {qualified_type_name(end)})"
        if eager and "Expr" in msg:
            msg = f"{msg}\n\nHint: Calling `nw.{name}` with expressions requires:\n"
            "  - `eager=False`\n"
            "  - a context such as `select` or `with_columns`"

        raise InvalidOperationError(msg)

    def try_unwrap_literals(self, node: RangeExpr[Self]) -> tuple[T_co, T_co] | None:
        """If we were passed `(lit, lit)`, return the wrapped python literals.

        Otherwise, the inputs for `node` must be evaluated in a backend.
        """
        from narwhals._plan.expressions import Lit

        start, end = node.input
        if isinstance(start, Lit) and isinstance(end, Lit):
            return self.ensure_py_scalars(start.value, end.value)
        return None


class IntRange(RangeFunction[int], dtype=get_dtype()):
    __slots__ = ("step", "dtype")  # noqa: RUF023
    step: int
    dtype: IntegerType

    @classmethod
    def _valid_types(cls) -> tuple[type[int]]:
        return (int,)


class DateRange(RangeFunction[dt.date], dtype=dtm.DATE):
    __slots__ = ("interval", "closed")  # noqa: RUF023
    interval: int
    closed: ClosedInterval

    @classmethod
    def _valid_types(cls) -> tuple[type[dt.date]]:
        return (dt.date,)


class LinearSpace(RangeFunction["int | float"], dtype=map_all(dtm.floats_dtype)):
    __slots__ = ("num_samples", "closed")  # noqa: RUF023
    num_samples: int
    closed: ClosedInterval

    @classmethod
    def _valid_types(cls) -> tuple[type[int | float], ...]:
        return (int, float)
