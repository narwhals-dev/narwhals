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
    from collections.abc import Iterable, Sequence

    from typing_extensions import TypeAlias, TypeIs

    from narwhals._compliant.selectors import CompliantSelectorNamespace
    from narwhals._utils import Implementation, Version
    from narwhals.dtypes import DType
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
    def _align_diagonal(
        self, frames: Sequence[CompliantFrameT], /
    ) -> Sequence[CompliantFrameT]:
        """Convert the inputs to `concat(..., how="diagonal")` into `concat(..., how="vertical")`.

        Adapted from [`convert_diagonal_concat`].

        [`convert_diagonal_concat`]: https://github.com/pola-rs/polars/blob/c2412600210a21143835c9dfcb0a9182f462b619/crates/polars-plan/src/plans/conversion/dsl_to_ir/concat.rs#L10-L68
        """
        schemas = [frame.collect_schema() for frame in frames]
        it_schemas = iter(schemas)
        total_schema = dict(next(it_schemas))
        seen_names = set(total_schema)
        to_add_fields: dict[str, DType] = {}
        for sch in it_schemas:
            to_add_fields.update(
                {name: dtype for name, dtype in sch.items() if name not in seen_names}
            )
            seen_names.update(to_add_fields)
        if not seen_names:
            return frames
        total_schema.update(to_add_fields)
        total_names = tuple(total_schema)
        added_exprs: dict[str, CompliantExprT_co] = {}
        results: list[CompliantFrameT] = []
        for frame, schema in zip(frames, schemas):
            to_add_exprs: list[CompliantExprT_co] = []
            for name, dtype in total_schema.items():
                if name not in schema:
                    maybe_seen = added_exprs.get(name)
                    if maybe_seen is None:
                        to_add_expr = self.lit(None, dtype).alias(name)
                        to_add_exprs.append(to_add_expr)
                        added_exprs[name] = to_add_expr
                    else:
                        to_add_exprs.append(maybe_seen)
            result = frame
            if to_add_exprs:
                result = result.with_columns(*to_add_exprs)
            results.append(result.simple_select(*total_names))
        return results


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
        when: NativeSeriesT,
        then: NativeSeriesT,
        otherwise: NativeSeriesT | None = None,
    ) -> NativeSeriesT: ...
    def when_then(
        self, predicate: EagerExprT, then: EagerExprT, otherwise: EagerExprT | None = None
    ) -> EagerExprT:
        def func(df: EagerDataFrameT) -> Sequence[EagerSeriesT_co]:
            predicate_s = df._evaluate_single_output_expr(predicate)
            align = predicate_s._align_full_broadcast

            then_s = df._evaluate_single_output_expr(then)
            if otherwise is None:
                predicate_s, then_s = align(predicate_s, then_s)
                result = self._if_then_else(predicate_s.native, then_s.native)

            if otherwise is None:
                predicate_s, then_s = align(predicate_s, then_s)
                result = self._if_then_else(predicate_s.native, then_s.native)
            else:
                otherwise_s = df._evaluate_single_output_expr(otherwise)
                predicate_s, then_s, otherwise_s = align(predicate_s, then_s, otherwise_s)
                result = self._if_then_else(
                    predicate_s.native, then_s.native, otherwise_s.native
                )
            return [then_s._with_native(result)]

        return self._expr._from_callable(
            func=func,
            evaluate_output_names=getattr(
                then, "_evaluate_output_names", lambda _df: ["literal"]
            ),
            alias_output_names=getattr(then, "_alias_output_names", None),
            context=predicate,
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
