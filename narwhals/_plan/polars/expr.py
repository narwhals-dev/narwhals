from __future__ import annotations

from typing import TYPE_CHECKING, Any, ClassVar

import polars as pl

from narwhals._plan.common import todo
from narwhals._plan.compliant.expr import CompliantExpr
from narwhals._plan.polars.namespace import PolarsNamespace, dtype_to_native
from narwhals._utils import Version

if TYPE_CHECKING:
    from typing_extensions import Self, TypeAlias

    from narwhals._plan import expressions as ir
    from narwhals._plan.polars.dataframe import PolarsDataFrame as DataFrame
    from narwhals.typing import IntoDType, PythonLiteral


Incomplete: TypeAlias = Any


class PolarsExpr(CompliantExpr["DataFrame", pl.Expr, pl.Expr]):
    __slots__ = ("_native",)
    _native: pl.Expr
    version: ClassVar = Version.MAIN

    def _with_native(self, native: pl.Expr, name: str = "", /) -> Self:
        return self.from_native(native, name)

    # NOTE: Unsure how much of `name` might be needed for polars
    @classmethod
    def from_native(cls, native: pl.Expr, name: str = "", /) -> Self:
        obj = cls.__new__(cls)
        obj._native = native if not name else native.alias(name)
        return obj

    @classmethod
    def from_python(
        cls,
        value: PythonLiteral,
        name: str = "literal",
        /,
        *,
        dtype: IntoDType | None = None,
    ) -> Self:
        unknown = cls.version.dtypes.Unknown
        dtype_pl = None if dtype == unknown else dtype_to_native(dtype, cls.version)
        return cls.from_native(pl.lit(value, dtype_pl), name)

    @property
    def native(self) -> pl.Expr:
        return self._native

    def __narwhals_namespace__(self) -> PolarsNamespace:
        return PolarsNamespace()

    @classmethod
    def from_ir(cls, node: ir.ExprIR, frame: DataFrame, name: str) -> PolarsExpr:
        obj = cls.__new__(cls)
        return node.dispatch(obj, frame, name)

    @classmethod
    def from_named_ir(cls, named_ir: ir.NamedIR, frame: Incomplete) -> PolarsExpr:
        return cls.from_ir(named_ir.expr, frame, named_ir.name)

    @classmethod
    def lit(cls, node: ir.Lit[PythonLiteral], _: Incomplete, name: str, /) -> Self:
        return cls.from_python(node.value, name, dtype=node.dtype)

    @classmethod
    def lit_series(
        cls, node: ir.LitSeries[pl.Series], _: Incomplete, name: str, /
    ) -> Self:
        return cls.from_native(pl.lit(node.native), name)

    @classmethod
    def len_star(cls, _: ir.Len, __: Incomplete, name: str, /) -> Self:
        return cls.from_native(pl.len(), name)

    abs = todo()
    all = todo()
    any = todo()
    arg_max = todo()
    arg_min = todo()
    binary_expr = todo()

    def cast(self, node: ir.Cast, frame: Incomplete, name: str) -> Self:
        dtype = dtype_to_native(node.dtype, self.version)
        return self._with_native(node.expr.dispatch(self, frame, name).native.cast(dtype))

    ceil = todo()
    clip = todo()
    clip_lower = todo()
    clip_upper = todo()
    count = todo()
    cum_count = todo()
    cum_max = todo()
    cum_min = todo()
    cum_prod = todo()
    cum_sum = todo()
    diff = todo()
    drop_nulls = todo()
    ewm_mean = todo()
    exp = todo()
    fill_nan = todo()
    fill_null = todo()
    fill_null_with_strategy = todo()
    filter = todo()

    # TODO @dangotbanned: Make `CompliantScalar` more optional
    # Return type here should be fine, but need a way to communicate that scalar-ness is handled elsewhere
    def first(self, node: ir.aggregation.First, frame: Any, name: str) -> Self:  # type: ignore[override]
        return self._with_native(node.expr.dispatch(self, frame, name).native.first())

    floor = todo()
    hist_bin_count = todo()
    hist_bins = todo()
    is_between = todo()
    is_duplicated = todo()
    is_finite = todo()
    is_first_distinct = todo()
    is_in_expr = todo()
    is_in_seq = todo()
    is_last_distinct = todo()
    is_nan = todo()
    is_not_nan = todo()
    is_not_null = todo()
    is_null = todo()
    is_unique = todo()
    kurtosis = todo()
    last = todo()
    len = todo()
    log = todo()
    max = todo()
    mean = todo()
    median = todo()
    min = todo()
    mode_all = todo()
    mode_any = todo()
    n_unique = todo()
    not_ = todo()
    null_count = todo()
    over = todo()
    over_ordered = todo()
    pow = todo()
    quantile = todo()
    rank = todo()
    replace_strict = todo()
    replace_strict_default = todo()
    rolling_sum = todo()
    rolling_mean = todo()
    rolling_std = todo()
    rolling_var = todo()
    round = todo()
    shift = todo()
    skew = todo()
    sort = todo()
    sort_by = todo()
    sqrt = todo()
    std = todo()
    sum = todo()
    ternary_expr = todo()
    unique = todo()
    var = todo()

    cat = todo()  # pyright: ignore[reportAssignmentType, reportIncompatibleMethodOverride]
    dt = todo()  # pyright: ignore[reportAssignmentType, reportIncompatibleMethodOverride]
    list = todo()  # pyright: ignore[reportAssignmentType, reportIncompatibleMethodOverride]
    str = todo()  # pyright: ignore[reportAssignmentType, reportIncompatibleMethodOverride]
    struct = todo()  # pyright: ignore[reportAssignmentType, reportIncompatibleMethodOverride]


PolarsExpr()
