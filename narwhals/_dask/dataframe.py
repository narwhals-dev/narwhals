from __future__ import annotations

from typing import TYPE_CHECKING
from typing import Any
from typing import Iterable
from typing import Literal
from typing import Sequence

from narwhals._dask.utils import parse_exprs_and_named_exprs
from narwhals._pandas_like.utils import translate_dtype
from narwhals.dependencies import get_dask_dataframe
from narwhals.dependencies import get_pandas
from narwhals.utils import Implementation
from narwhals.utils import flatten
from narwhals.utils import generate_unique_token
from narwhals.utils import parse_version

if TYPE_CHECKING:
    from typing_extensions import Self

    from narwhals._dask.expr import DaskExpr
    from narwhals._dask.namespace import DaskNamespace
    from narwhals._dask.typing import IntoDaskExpr
    from narwhals.dtypes import DType


class DaskLazyFrame:
    def __init__(
        self, native_dataframe: Any, *, backend_version: tuple[int, ...]
    ) -> None:
        self._native_dataframe = native_dataframe
        self._backend_version = backend_version

    def __native_namespace__(self) -> Any:  # pragma: no cover
        return get_dask_dataframe()

    def __narwhals_namespace__(self) -> DaskNamespace:
        from narwhals._dask.namespace import DaskNamespace

        return DaskNamespace(backend_version=self._backend_version)

    def __narwhals_lazyframe__(self) -> Self:
        return self

    def _from_native_dataframe(self, df: Any) -> Self:
        return self.__class__(df, backend_version=self._backend_version)

    def with_columns(self, *exprs: DaskExpr, **named_exprs: DaskExpr) -> Self:
        df = self._native_dataframe
        new_series = parse_exprs_and_named_exprs(self, *exprs, **named_exprs)
        df = df.assign(**new_series)
        return self._from_native_dataframe(df)

    def collect(self) -> Any:
        from narwhals._pandas_like.dataframe import PandasLikeDataFrame

        result = self._native_dataframe.compute()
        return PandasLikeDataFrame(
            result,
            implementation=Implementation.PANDAS,
            backend_version=parse_version(get_pandas().__version__),
        )

    @property
    def columns(self) -> list[str]:
        return self._native_dataframe.columns.tolist()  # type: ignore[no-any-return]

    def filter(
        self,
        *predicates: DaskExpr,
    ) -> Self:
        from narwhals._dask.namespace import DaskNamespace

        plx = DaskNamespace(backend_version=self._backend_version)
        expr = plx.all_horizontal(*predicates)
        # Safety: all_horizontal's expression only returns a single column.
        mask = expr._call(self)[0]
        return self._from_native_dataframe(self._native_dataframe.loc[mask])

    def lazy(self) -> Self:
        return self

    def select(
        self: Self,
        *exprs: IntoDaskExpr,
        **named_exprs: IntoDaskExpr,
    ) -> Self:
        dd = get_dask_dataframe()

        if exprs and all(isinstance(x, str) for x in exprs) and not named_exprs:
            # This is a simple slice => fastpath!
            return self._from_native_dataframe(self._native_dataframe.loc[:, exprs])

        new_series = parse_exprs_and_named_exprs(self, *exprs, **named_exprs)

        if not new_series:
            # return empty dataframe, like Polars does
            pd = get_pandas()
            return self._from_native_dataframe(dd.from_pandas(pd.DataFrame()))

        if all(getattr(expr, "_returns_scalar", False) for expr in exprs) and all(
            getattr(val, "_returns_scalar", False) for val in named_exprs.values()
        ):
            df = dd.concat(
                [val.to_series().rename(name) for name, val in new_series.items()], axis=1
            )
            return self._from_native_dataframe(df)

        df = self._native_dataframe.assign(**new_series).loc[:, list(new_series.keys())]
        return self._from_native_dataframe(df)

    def drop_nulls(self) -> Self:
        return self._from_native_dataframe(self._native_dataframe.dropna())

    @property
    def schema(self) -> dict[str, DType]:
        return {
            col: translate_dtype(self._native_dataframe.loc[:, col])
            for col in self._native_dataframe.columns
        }

    def collect_schema(self) -> dict[str, DType]:
        return self.schema

    def drop(self: Self, *columns: str) -> Self:
        return self._from_native_dataframe(
            self._native_dataframe.drop(columns=list(columns))
        )

    def with_row_index(self: Self, name: str) -> Self:
        # Implementation is based on the following StackOverflow reply:
        # https://stackoverflow.com/questions/60831518/in-dask-how-does-one-add-a-range-of-integersauto-increment-to-a-new-column/60852409#60852409
        return self._from_native_dataframe(
            self._native_dataframe.assign(**{name: 1}).assign(
                **{name: lambda t: t[name].cumsum(method="blelloch") - 1}
            )
        )

    def rename(self: Self, mapping: dict[str, str]) -> Self:
        return self._from_native_dataframe(self._native_dataframe.rename(columns=mapping))

    def unique(
        self: Self,
        subset: str | list[str] | None,
        *,
        keep: Literal["any", "first", "last", "none"] = "any",
        maintain_order: bool = False,
    ) -> Self:
        """
        NOTE:
            The param `maintain_order` is only here for compatibility with the polars API
            and has no effect on the output.
        """
        subset = flatten(subset) if subset else None
        native_frame = self._native_dataframe
        if keep == "none":
            subset = subset or self.columns
            token = generate_unique_token(n_bytes=8, columns=subset)
            unique = (
                native_frame.groupby(subset)
                .size()
                .rename(token)
                .loc[lambda t: t == 1]
                .reset_index()
                .drop(columns=token)
            )
            result = native_frame.merge(unique, on=subset, how="inner")
        else:
            mapped_keep = {"any": "first"}.get(keep, keep)
            result = native_frame.drop_duplicates(subset=subset, keep=mapped_keep)
        return self._from_native_dataframe(result)

    def sort(
        self: Self,
        by: str | Iterable[str],
        *more_by: str,
        descending: bool | Sequence[bool] = False,
    ) -> Self:
        flat_keys = flatten([*flatten([by]), *more_by])
        df = self._native_dataframe
        if isinstance(descending, bool):
            ascending: bool | list[bool] = not descending
        else:
            ascending = [not d for d in descending]
        return self._from_native_dataframe(df.sort_values(flat_keys, ascending=ascending))

    def join(
        self: Self,
        other: Self,
        *,
        how: Literal["left", "inner", "outer", "cross", "anti", "semi"] = "inner",
        left_on: str | list[str] | None,
        right_on: str | list[str] | None,
    ) -> Self:
        if isinstance(left_on, str):
            left_on = [left_on]
        if isinstance(right_on, str):
            right_on = [right_on]

        if how == "cross":
            key_token = generate_unique_token(
                n_bytes=8, columns=[*self.columns, *other.columns]
            )

            return self._from_native_dataframe(
                self._native_dataframe.assign(**{key_token: 0}).merge(
                    other._native_dataframe.assign(**{key_token: 0}),
                    how="inner",
                    left_on=key_token,
                    right_on=key_token,
                    suffixes=("", "_right"),
                ),
            ).drop(key_token)

        if how == "anti":
            indicator_token = generate_unique_token(
                n_bytes=8, columns=[*self.columns, *other.columns]
            )

            other_native = (
                other._native_dataframe.loc[:, right_on]
                .rename(  # rename to avoid creating extra columns in join
                    columns=dict(zip(right_on, left_on))  # type: ignore[arg-type]
                )
                .drop_duplicates()
            )
            return self._from_native_dataframe(
                self._native_dataframe.merge(
                    other_native,
                    how="outer",
                    indicator=indicator_token,
                    left_on=left_on,
                    right_on=left_on,
                )
                .loc[lambda t: t[indicator_token] == "left_only"]
                .drop(columns=[indicator_token])
            )

        if how == "semi":
            other_native = (
                other._native_dataframe.loc[:, right_on]
                .rename(  # rename to avoid creating extra columns in join
                    columns=dict(zip(right_on, left_on))  # type: ignore[arg-type]
                )
                .drop_duplicates()  # avoids potential rows duplication from inner join
            )
            return self._from_native_dataframe(
                self._native_dataframe.merge(
                    other_native,
                    how="inner",
                    left_on=left_on,
                    right_on=left_on,
                )
            )

        if how == "left":
            other_native = other._native_dataframe
            result_native = self._native_dataframe.merge(
                other_native,
                how="left",
                left_on=left_on,
                right_on=right_on,
                suffixes=("", "_right"),
            )
            extra = []
            for left_key, right_key in zip(left_on, right_on):  # type: ignore[arg-type]
                if right_key != left_key and right_key not in self.columns:
                    extra.append(right_key)
                elif right_key != left_key:
                    extra.append(f"{right_key}_right")
            return self._from_native_dataframe(result_native.drop(columns=extra))

        return self._from_native_dataframe(
            self._native_dataframe.merge(
                other._native_dataframe,
                left_on=left_on,
                right_on=right_on,
                how=how,
                suffixes=("", "_right"),
            ),
        )

    def group_by(self, *by: str) -> Any:
        from narwhals._dask.group_by import DaskLazyGroupBy

        return DaskLazyGroupBy(self, list(by))
