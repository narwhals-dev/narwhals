from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol

from narwhals._plan.compliant.expr import CompliantExpr, EagerExpr, LazyExpr
from narwhals._plan.compliant.typing import FrameT_contra, LengthT, SeriesT, SeriesT_co
from narwhals._utils import not_implemented

if TYPE_CHECKING:
    from typing_extensions import Self

    from narwhals._plan import expressions as ir
    from narwhals._plan.expressions import FunctionExpr, aggregation as agg
    from narwhals._plan.expressions.functions import EwmMean, NullCount, Shift
    from narwhals._utils import Version
    from narwhals.typing import IntoDType, PythonLiteral


class CompliantScalar(
    CompliantExpr[FrameT_contra, SeriesT_co], Protocol[FrameT_contra, SeriesT_co]
):
    _name: str

    def _cast_float(self, node: ir.ExprIR, frame: FrameT_contra, name: str) -> Self:
        """`polars` interpolates a single scalar as a float."""
        dtype = self.version.dtypes.Float64()
        return self.cast(node.cast(dtype), frame, name)

    def _with_evaluated(self, evaluated: Any, name: str) -> Self:
        """Expr is based on a series having these via accessors, but a scalar needs to keep passing through."""
        cls = type(self)
        obj = cls.__new__(cls)
        obj._evaluated = evaluated
        obj._name = name or self.name
        obj._version = self.version
        return obj

    # NOTE: Constant behaviors with scalars observed in `polars`

    def _always_nan(self, node: ir.ExprIR, frame: Any, name: str) -> Self:
        return self.from_python(float("nan"), name, dtype=None, version=self.version)

    def _always_noop(self, node: ir.ExprIR, frame: Any, name: str) -> Self:
        return self._with_evaluated(self._evaluated, name)

    def _always_true(self, node: ir.ExprIR, frame: Any, name: str) -> Self:
        return self.from_python(True, name, dtype=None, version=self.version)

    def _always_false(self, node: ir.ExprIR, frame: Any, name: str) -> Self:
        return self.from_python(False, name, dtype=None, version=self.version)

    def _always_null(self, node: ir.ExprIR, frame: Any, name: str) -> Self:
        return self.from_python(None, name, dtype=None, version=self.version)

    def _always_zero(self, node: ir.ExprIR, frame: Any, name: str) -> Self:
        return self.from_python(0, name, dtype=None, version=self.version)

    def _always_one(self, node: ir.ExprIR, frame: Any, name: str) -> Self:
        return self.from_python(1, name, dtype=None, version=self.version)

    @property
    def name(self) -> str:
        return self._name

    @classmethod
    def from_python(
        cls,
        value: PythonLiteral,
        name: str = "literal",
        /,
        *,
        dtype: IntoDType | None,
        version: Version,
    ) -> Self: ...

    def count(self, node: agg.Count, frame: FrameT_contra, name: str) -> Self:
        """Returns 0 if null, else 1."""
        ...

    def ewm_mean(
        self, node: FunctionExpr[EwmMean], frame: FrameT_contra, name: str
    ) -> Self:
        return self._cast_float(node.input[0], frame, name)

    def mean(self, node: agg.Mean, frame: FrameT_contra, name: str) -> Self:
        return self._cast_float(node.expr, frame, name)

    def median(self, node: agg.Median, frame: FrameT_contra, name: str) -> Self:
        return self._cast_float(node.expr, frame, name)

    def null_count(
        self, node: FunctionExpr[NullCount], frame: FrameT_contra, name: str
    ) -> Self:
        """Returns 1 if null, else 0."""
        ...

    def quantile(self, node: agg.Quantile, frame: FrameT_contra, name: str) -> Self:
        return self._cast_float(node.expr, frame, name)

    def shift(self, node: FunctionExpr[Shift], frame: FrameT_contra, name: str) -> Self:
        if node.function.n == 0:
            return self._with_evaluated(self._evaluated, name)
        return self.from_python(None, name, dtype=None, version=self.version)

    arg_max = _always_zero  # type: ignore[misc]
    arg_min = _always_zero  # type: ignore[misc]
    is_first_distinct = _always_true  # type: ignore[misc]
    is_last_distinct = _always_true  # type: ignore[misc]
    is_unique = _always_true  # type: ignore[misc]
    is_duplicated = _always_false  # type: ignore[misc]
    n_unique = _always_one  # type: ignore[misc]
    std = _always_null  # type: ignore[misc]
    var = _always_null  # type: ignore[misc]
    first = _always_noop  # type: ignore[misc]
    max = _always_noop  # type: ignore[misc]
    min = _always_noop  # type: ignore[misc]
    last = _always_noop  # type: ignore[misc]
    sort = _always_noop  # type: ignore[misc]
    sort_by = _always_noop  # type: ignore[misc]
    sum = _always_noop  # type: ignore[misc]
    mode = _always_noop  # type: ignore[misc]
    unique = _always_noop  # type: ignore[misc]
    kurtosis = _always_nan  # type: ignore[misc]
    skew = _always_nan  # type: ignore[misc]
    fill_null_with_strategy = not_implemented()  # type: ignore[misc]
    hist_bins = not_implemented()  # type: ignore[misc]
    hist_bin_count = not_implemented()  # type: ignore[misc]
    len = _always_one  # type: ignore[misc]


class EagerScalar(
    CompliantScalar[FrameT_contra, SeriesT],
    EagerExpr[FrameT_contra, SeriesT],
    Protocol[FrameT_contra, SeriesT],
):
    def __len__(self) -> int:
        return 1

    def to_python(self) -> PythonLiteral: ...

    gather_every = not_implemented()  # type: ignore[misc]


class LazyScalar(
    CompliantScalar[FrameT_contra, SeriesT],
    LazyExpr[FrameT_contra, SeriesT, LengthT],
    Protocol[FrameT_contra, SeriesT, LengthT],
): ...
