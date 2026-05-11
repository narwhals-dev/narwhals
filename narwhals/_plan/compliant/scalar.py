from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol

from narwhals._plan.compliant.expr import CompliantColumn, CompliantExpr
from narwhals._plan.compliant.typing import (
    DeprecatedFrameT_contra as Frame,
    NativeExpr_co,
    NativeScalar_co,
)
from narwhals._utils import Version

if TYPE_CHECKING:
    from collections.abc import Callable

    from typing_extensions import Self

    from narwhals._plan import expressions as ir
    from narwhals._plan.expressions import (
        FunctionExpr as FExpr,
        aggregation as agg,
        functions as F,
    )
    from narwhals.typing import IntoDType, PythonLiteral

_F64 = Version.MAIN.dtypes.Float64()


class CompliantScalar(
    CompliantColumn[Frame, NativeScalar_co, NativeExpr_co, NativeScalar_co],
    Protocol[Frame, NativeExpr_co, NativeScalar_co],
):
    """`[FrameT_contra, NativeExpr_co, NativeScalar_co]`."""

    __slots__ = ("_evaluated", "_name")

    _evaluated: Any
    """Compliant or native value.

    - `ArrowExpr` uses `ArrowSeries`
    - `ArrowScalar` uses `pa.Scalar[Any]`
    - `PolarsExpr` uses `pl.Expr`
    """
    _name: str

    @property
    def name(self) -> str:
        return self._name

    def _cast_float(self, node: ir.ExprIR, frame: Frame, name: str) -> Self:
        """`polars` interpolates a single scalar as a float."""
        return self.cast(node.cast(_F64), frame, name)

    def _with_evaluated(self, evaluated: Any, name: str) -> Self:
        """Expr is based on a series having these via accessors, but a scalar needs to keep passing through."""
        cls = type(self)
        obj = cls.__new__(cls)
        obj._evaluated = evaluated
        obj._name = name or self.name
        return obj

    # NOTE: Constant behaviors with scalars observed in `polars`

    def _always_nan(self, node: ir.ExprIR, frame: Any, name: str, /) -> Self:
        return self.from_python(float("nan"), name, dtype=None)

    def _always_noop(self, node: ir.ExprIR, frame: Any, name: str, /) -> Self:
        return self._with_evaluated(self._evaluated, name)

    def _always_true(self, node: ir.ExprIR, frame: Any, name: str, /) -> Self:
        return self.from_python(True, name, dtype=None)

    def _always_false(self, node: ir.ExprIR, frame: Any, name: str, /) -> Self:
        return self.from_python(False, name, dtype=None)

    def _always_null(self, node: ir.ExprIR, frame: Any, name: str, /) -> Self:
        return self.from_python(None, name, dtype=None)

    def _always_zero(self, node: ir.ExprIR, frame: Any, name: str, /) -> Self:
        return self.from_python(0, name, dtype=None)

    def _always_one(self, node: ir.ExprIR, frame: Any, name: str, /) -> Self:
        return self.from_python(1, name, dtype=None)

    @classmethod
    def lit(cls, node: ir.Lit[PythonLiteral], _: Any, name: str, /) -> Self:
        return cls.from_python(node.value, name, dtype=node.dtype)

    @classmethod
    def from_python(
        cls, value: PythonLiteral, name: str = "literal", /, *, dtype: IntoDType | None
    ) -> Self: ...

    def count(
        self, node: agg.Count, frame: Frame, name: str, /
    ) -> CompliantScalar[Frame, NativeExpr_co, NativeScalar_co]:
        """Returns 0 if null, else 1."""
        ...

    def ewm_mean(
        self, node: FExpr[F.EwmMean], frame: Frame, name: str, /
    ) -> CompliantScalar[Frame, NativeExpr_co, NativeScalar_co]:
        return self._cast_float(node.input[0], frame, name)

    def mean(
        self, node: agg.Mean, frame: Frame, name: str, /
    ) -> CompliantScalar[Frame, NativeExpr_co, NativeScalar_co]:
        return self._cast_float(node.expr, frame, name)

    def median(
        self, node: agg.Median, frame: Frame, name: str, /
    ) -> CompliantScalar[Frame, NativeExpr_co, NativeScalar_co]:
        return self._cast_float(node.expr, frame, name)

    def null_count(
        self, node: FExpr[F.NullCount], frame: Frame, name: str
    ) -> CompliantScalar[Frame, NativeExpr_co, NativeScalar_co]:
        """Returns 1 if null, else 0."""
        ...

    def quantile(
        self, node: agg.Quantile, frame: Frame, name: str, /
    ) -> CompliantScalar[Frame, NativeExpr_co, NativeScalar_co]:
        return self._cast_float(node.expr, frame, name)

    def shift(
        self, node: FExpr[F.Shift], frame: Frame, name: str, /
    ) -> CompliantScalar[Frame, NativeExpr_co, NativeScalar_co]:
        if node.function.n == 0:
            return self._with_evaluated(self._evaluated, name)
        return self.from_python(None, name, dtype=None)

    def drop_nulls(
        self, node: FExpr[F.DropNulls], frame: Frame, name: str, /
    ) -> (
        CompliantScalar[Frame, NativeExpr_co, NativeScalar_co]
        | CompliantExpr[Frame, NativeExpr_co, NativeScalar_co]
    ):
        """Returns a 0-length Series if null, else noop."""
        ...

    arg_max: Callable[..., Self] = _always_zero
    arg_min: Callable[..., Self] = _always_zero
    is_first_distinct: Callable[..., Self] = _always_true
    is_last_distinct: Callable[..., Self] = _always_true
    is_unique: Callable[..., Self] = _always_true
    is_duplicated: Callable[..., Self] = _always_false
    n_unique: Callable[..., Self] = _always_one
    std: Callable[..., Self] = _always_null
    var: Callable[..., Self] = _always_null
    first: Callable[..., Self] = _always_noop
    max: Callable[..., Self] = _always_noop
    min: Callable[..., Self] = _always_noop
    last: Callable[..., Self] = _always_noop
    sort: Callable[..., Self] = _always_noop
    sort_by: Callable[..., Self] = _always_noop
    sum: Callable[..., Self] = _always_noop
    mode: Callable[..., Self] = _always_noop
    unique: Callable[..., Self] = _always_noop
    mode_any: Callable[..., Self] = _always_noop
    kurtosis: Callable[..., Self] = _always_nan
    skew: Callable[..., Self] = _always_nan
    len: Callable[..., Self] = _always_one


class EagerScalar(
    CompliantScalar[Frame, NativeExpr_co, NativeScalar_co],
    Protocol[Frame, NativeExpr_co, NativeScalar_co],
):
    """`[FrameT_contra, NativeExpr_co, NativeScalar_co]`."""

    __slots__ = ()

    def to_python(self) -> PythonLiteral: ...
    def __len__(self) -> int:
        return 1
