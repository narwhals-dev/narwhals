from __future__ import annotations

from functools import partial
from typing import TYPE_CHECKING, Any, Protocol, overload

from narwhals._compliant.typing import (
    CompliantExprT,
    CompliantExprT_co,
    CompliantFrameT,
    CompliantLazyFrameT,
    DepthTrackingExprT,
    EagerDataFrameT,
    EagerExprT,
    EagerSeriesT_co,
    LazyExprT,
    NativeFrameT,
    NativeSeriesT,
)
from narwhals._utils import (
    exclude_column_names,
    get_column_names,
    passthrough_column_names,
)
from narwhals.dependencies import is_numpy_array_2d

if TYPE_CHECKING:
    from collections.abc import Collection, Iterable, Iterator, KeysView, Sequence

    from typing_extensions import TypeAlias, TypeIs

    from narwhals._compliant.selectors import CompliantSelectorNamespace
    from narwhals._utils import Implementation, Version
    from narwhals.typing import (
        ConcatMethod,
        Into1DArray,
        IntoDType,
        IntoSchema,
        NonNestedLiteral,
        _2DArray,
    )

    Incomplete: TypeAlias = Any

__all__ = [
    "CompliantNamespace",
    "DepthTrackingNamespace",
    "EagerNamespace",
    "LazyNamespace",
]


class CompliantNamespace(Protocol[CompliantFrameT, CompliantExprT]):
    # NOTE: `narwhals`
    _implementation: Implementation
    _version: Version

    @property
    def _expr(self) -> type[CompliantExprT]: ...
    # NOTE: `polars`
    def all(self) -> CompliantExprT:
        return self._expr.from_column_names(get_column_names, context=self)

    def col(self, *names: str) -> CompliantExprT:
        return self._expr.from_column_names(passthrough_column_names(names), context=self)

    def exclude(self, *names: str) -> CompliantExprT:
        return self._expr.from_column_names(
            partial(exclude_column_names, names=names), context=self
        )

    def nth(self, indices: Sequence[int]) -> CompliantExprT:
        return self._expr.from_column_indices(*indices, context=self)

    def len(self) -> CompliantExprT: ...
    def lit(self, value: NonNestedLiteral, dtype: IntoDType | None) -> CompliantExprT: ...
    def all_horizontal(
        self, *exprs: CompliantExprT, ignore_nulls: bool
    ) -> CompliantExprT: ...
    def any_horizontal(
        self, *exprs: CompliantExprT, ignore_nulls: bool
    ) -> CompliantExprT: ...
    def sum_horizontal(self, *exprs: CompliantExprT) -> CompliantExprT: ...
    def mean_horizontal(self, *exprs: CompliantExprT) -> CompliantExprT: ...
    def min_horizontal(self, *exprs: CompliantExprT) -> CompliantExprT: ...
    def max_horizontal(self, *exprs: CompliantExprT) -> CompliantExprT: ...
    def concat(
        self, items: Iterable[CompliantFrameT], *, how: ConcatMethod
    ) -> CompliantFrameT: ...
    def concat_str(
        self, *exprs: CompliantExprT, separator: str, ignore_nulls: bool
    ) -> CompliantExprT: ...
    @property
    def selectors(self) -> CompliantSelectorNamespace[Any, Any]: ...
    def coalesce(self, *exprs: CompliantExprT) -> CompliantExprT: ...
    # NOTE: typing this accurately requires 2x more `TypeVar`s
    def from_native(self, data: Any, /) -> Any: ...
    def is_native(self, obj: Any, /) -> TypeIs[Any]:
        """Return `True` if `obj` can be passed to `from_native`."""
        ...


class AlignDiagonal(Protocol[CompliantFrameT, CompliantExprT_co]):
    """Mixin to help support `"diagonal*"` concatenation."""

    def lit(
        self, value: NonNestedLiteral, dtype: IntoDType | None
    ) -> CompliantExprT_co: ...
    def align_diagonal(
        self, frames: Collection[CompliantFrameT], /
    ) -> Sequence[CompliantFrameT]:
        """Prepare frames with differing schemas for vertical concatenation.

        Adapted from [`convert_diagonal_concat`].

        [`convert_diagonal_concat`]: https://github.com/pola-rs/polars/blob/c2412600210a21143835c9dfcb0a9182f462b619/crates/polars-plan/src/plans/conversion/dsl_to_ir/concat.rs#L10-L68
        """
        schemas = [frame.collect_schema() for frame in frames]
        # 1 - Take the first schema, collecting any fields from the remaining that introduce new names
        # Subtle difference from `dict |= dict |= ...` as we preserve the first `DType`, rather than the last.
        it_schemas = iter(schemas)
        union = dict(next(it_schemas))
        seen = union.keys()
        for schema in it_schemas:
            union.update((nm, dtype) for nm, dtype in schema.items() if nm not in seen)
        return self._align_diagonal(frames, schemas, union)

    def _align_diagonal(
        self,
        frames: Iterable[CompliantFrameT],
        schemas: Iterable[IntoSchema],
        union_schema: IntoSchema,
    ) -> Sequence[CompliantFrameT]:
        union_names = union_schema.keys()
        # Lazily populate null expressions as needed, shared across frames
        null_exprs: dict[str, CompliantExprT_co] = {}

        def iter_missing_exprs(missing: Iterable[str]) -> Iterator[CompliantExprT_co]:
            nonlocal null_exprs
            for name in missing:
                if (expr := null_exprs.get(name)) is None:
                    dtype = union_schema[name]
                    expr = null_exprs[name] = self.lit(None, dtype).alias(name)
                yield expr

        def align(df: CompliantFrameT, columns: KeysView[str]) -> CompliantFrameT:
            if missing := union_names - columns:
                df = df.with_columns(*iter_missing_exprs(missing))
            # Even if all fields are present, we always reorder the columns to match between frames.
            return df.simple_select(*union_names)

        return [align(frame, schema.keys()) for frame, schema in zip(frames, schemas)]


class DepthTrackingNamespace(
    CompliantNamespace[CompliantFrameT, DepthTrackingExprT],
    Protocol[CompliantFrameT, DepthTrackingExprT],
):
    def all(self) -> DepthTrackingExprT:
        return self._expr.from_column_names(get_column_names, context=self)

    def col(self, *names: str) -> DepthTrackingExprT:
        return self._expr.from_column_names(passthrough_column_names(names), context=self)

    def exclude(self, *names: str) -> DepthTrackingExprT:
        return self._expr.from_column_names(
            partial(exclude_column_names, names=names), context=self
        )


class LazyNamespace(
    CompliantNamespace[CompliantLazyFrameT, LazyExprT],
    Protocol[CompliantLazyFrameT, LazyExprT, NativeFrameT],
):
    @property
    def _backend_version(self) -> tuple[int, ...]:
        return self._implementation._backend_version()

    @property
    def _lazyframe(self) -> type[CompliantLazyFrameT]: ...
    def is_native(self, obj: Any, /) -> TypeIs[NativeFrameT]:
        return self._lazyframe._is_native(obj)

    def from_native(self, data: NativeFrameT | Any, /) -> CompliantLazyFrameT:
        if self._lazyframe._is_native(data):
            return self._lazyframe.from_native(data, context=self)
        msg = f"Unsupported type: {type(data).__name__!r}"  # pragma: no cover
        raise TypeError(msg)


class EagerNamespace(
    DepthTrackingNamespace[EagerDataFrameT, EagerExprT],
    Protocol[EagerDataFrameT, EagerSeriesT_co, EagerExprT, NativeFrameT, NativeSeriesT],
):
    @property
    def _backend_version(self) -> tuple[int, ...]:
        return self._implementation._backend_version()

    @property
    def _dataframe(self) -> type[EagerDataFrameT]: ...
    @property
    def _series(self) -> type[EagerSeriesT_co]: ...
    def _if_then_else(
        self,
        when: NativeSeriesT | list[NativeSeriesT],
        then: NativeSeriesT | list[NativeSeriesT],
        otherwise: NativeSeriesT | None = None,
    ) -> NativeSeriesT: ...
    def _when_then_simple(
        self, df: EagerDataFrameT, predicate: EagerExprT, then: EagerExprT
    ) -> Sequence[EagerSeriesT_co]:
        predicate_s = df._evaluate_single_output_expr(predicate)
        then_s = df._evaluate_single_output_expr(then)
        predicate_s, then_s = predicate_s._align_full_broadcast(predicate_s, then_s)
        result = self._if_then_else(predicate_s.native, then_s.native)
        return [then_s._with_native(result)]

    def _when_then_chained(
        self, df: EagerDataFrameT, args: tuple[EagerExprT, ...]
    ) -> Sequence[EagerSeriesT_co]:
        evaluated = [df._evaluate_single_output_expr(arg) for arg in args]
        *pairs_list, otherwise_s = (
            evaluated if len(evaluated) % 2 == 1 else (*evaluated, None)
        )
        conditions = pairs_list[::2]
        values = pairs_list[1::2]

        all_series = (
            [*conditions, *values, otherwise_s]
            if otherwise_s is not None
            else [*conditions, *values]
        )
        aligned = conditions[0]._align_full_broadcast(*all_series)

        num_conditions = len(conditions)
        aligned_conditions = aligned[:num_conditions]
        aligned_values = aligned[num_conditions : num_conditions * 2]
        aligned_otherwise = aligned[-1] if otherwise_s is not None else None

        if len(conditions) == 1:
            result = self._if_then_else(
                aligned_conditions[0].native,
                aligned_values[0].native,
                aligned_otherwise.native if aligned_otherwise is not None else None,
            )
        else:
            result = self._if_then_else(
                [c.native for c in aligned_conditions],
                [v.native for v in aligned_values],
                aligned_otherwise.native if aligned_otherwise is not None else None,
            )

        return [values[0]._with_native(result)]

    def when_then(self, *args: EagerExprT) -> EagerExprT:
        def func(df: EagerDataFrameT) -> Sequence[EagerSeriesT_co]:
            if len(args) == 2:
                return self._when_then_simple(df, args[0], args[1])
            return self._when_then_chained(df, args)

        return self._expr._from_callable(
            func=func,
            evaluate_output_names=getattr(
                args[1], "_evaluate_output_names", lambda _df: ["literal"]
            ),
            alias_output_names=getattr(args[1], "_alias_output_names", None),
            context=args[0],
        )

    def is_native(self, obj: Any, /) -> TypeIs[NativeFrameT | NativeSeriesT]:
        return self._dataframe._is_native(obj) or self._series._is_native(obj)

    @overload
    def from_native(self, data: NativeFrameT, /) -> EagerDataFrameT: ...
    @overload
    def from_native(self, data: NativeSeriesT, /) -> EagerSeriesT_co: ...
    def from_native(
        self, data: NativeFrameT | NativeSeriesT | Any, /
    ) -> EagerDataFrameT | EagerSeriesT_co:
        if self._dataframe._is_native(data):
            return self._dataframe.from_native(data, context=self)
        if self._series._is_native(data):
            return self._series.from_native(data, context=self)
        msg = f"Unsupported type: {type(data).__name__!r}"
        raise TypeError(msg)

    @overload
    def from_numpy(self, data: Into1DArray, /, schema: None = ...) -> EagerSeriesT_co: ...

    @overload
    def from_numpy(
        self, data: _2DArray, /, schema: IntoSchema | Sequence[str] | None
    ) -> EagerDataFrameT: ...

    def from_numpy(
        self,
        data: Into1DArray | _2DArray,
        /,
        schema: IntoSchema | Sequence[str] | None = None,
    ) -> EagerDataFrameT | EagerSeriesT_co:
        if is_numpy_array_2d(data):
            return self._dataframe.from_numpy(data, schema=schema, context=self)
        return self._series.from_numpy(data, context=self)

    def _concat_diagonal(self, dfs: Sequence[NativeFrameT], /) -> NativeFrameT: ...
    def _concat_horizontal(
        self, dfs: Sequence[NativeFrameT | Any], /
    ) -> NativeFrameT: ...
    def _concat_vertical(self, dfs: Sequence[NativeFrameT], /) -> NativeFrameT: ...
    def concat(
        self, items: Iterable[EagerDataFrameT], *, how: ConcatMethod
    ) -> EagerDataFrameT:
        dfs = [item.native for item in items]
        if how == "horizontal":
            native = self._concat_horizontal(dfs)
        elif how == "vertical":
            native = self._concat_vertical(dfs)
        elif how == "diagonal":
            native = self._concat_diagonal(dfs)
        else:  # pragma: no cover
            raise NotImplementedError
        return self._dataframe.from_native(native, context=self)
