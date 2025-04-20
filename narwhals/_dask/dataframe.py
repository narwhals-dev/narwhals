from __future__ import annotations

from typing import TYPE_CHECKING
from typing import Any
from typing import Iterator
from typing import Mapping
from typing import Sequence

import dask.dataframe as dd
import pandas as pd

from narwhals._dask.utils import add_row_index
from narwhals._dask.utils import evaluate_exprs
from narwhals._pandas_like.utils import native_to_narwhals_dtype
from narwhals._pandas_like.utils import select_columns_by_name
from narwhals.typing import CompliantDataFrame
from narwhals.typing import CompliantLazyFrame
from narwhals.utils import Implementation
from narwhals.utils import _remap_full_join_keys
from narwhals.utils import check_column_exists
from narwhals.utils import check_column_names_are_unique
from narwhals.utils import generate_temporary_column_name
from narwhals.utils import not_implemented
from narwhals.utils import parse_columns_to_drop
from narwhals.utils import parse_version
from narwhals.utils import validate_backend_version

if TYPE_CHECKING:
    from types import ModuleType

    import dask.dataframe.dask_expr as dx
    from typing_extensions import Self
    from typing_extensions import TypeIs

    from narwhals._dask.expr import DaskExpr
    from narwhals._dask.group_by import DaskLazyGroupBy
    from narwhals._dask.namespace import DaskNamespace
    from narwhals.dtypes import DType
    from narwhals.typing import AsofJoinStrategy
    from narwhals.typing import JoinStrategy
    from narwhals.typing import LazyUniqueKeepStrategy
    from narwhals.utils import Version
    from narwhals.utils import _FullContext


class DaskLazyFrame(CompliantLazyFrame["DaskExpr", "dd.DataFrame"]):
    def __init__(
        self,
        native_dataframe: dd.DataFrame,
        *,
        backend_version: tuple[int, ...],
        version: Version,
    ) -> None:
        self._native_frame: dd.DataFrame = native_dataframe
        self._backend_version = backend_version
        self._implementation = Implementation.DASK
        self._version = version
        self._cached_schema: dict[str, DType] | None = None
        self._cached_columns: list[str] | None = None
        validate_backend_version(self._implementation, self._backend_version)

    @staticmethod
    def _is_native(obj: dd.DataFrame | Any) -> TypeIs[dd.DataFrame]:
        return isinstance(obj, dd.DataFrame)

    @classmethod
    def from_native(cls, data: dd.DataFrame, /, *, context: _FullContext) -> Self:
        return cls(
            data, backend_version=context._backend_version, version=context._version
        )

    def __native_namespace__(self) -> ModuleType:
        if self._implementation is Implementation.DASK:
            return self._implementation.to_native_namespace()

        msg = f"Expected dask, got: {type(self._implementation)}"  # pragma: no cover
        raise AssertionError(msg)

    def __narwhals_namespace__(self) -> DaskNamespace:
        from narwhals._dask.namespace import DaskNamespace

        return DaskNamespace(backend_version=self._backend_version, version=self._version)

    def __narwhals_lazyframe__(self) -> Self:
        return self

    def _with_version(self, version: Version) -> Self:
        return self.__class__(
            self.native, backend_version=self._backend_version, version=version
        )

    def _with_native(self, df: Any) -> Self:
        return self.__class__(
            df, backend_version=self._backend_version, version=self._version
        )

    def _iter_columns(self) -> Iterator[dx.Series]:
        for _col, ser in self.native.items():  # noqa: PERF102
            yield ser

    def with_columns(self, *exprs: DaskExpr) -> Self:
        new_series = evaluate_exprs(self, *exprs)
        return self._with_native(self.native.assign(**dict(new_series)))

    def collect(
        self,
        backend: Implementation | None,
        **kwargs: Any,
    ) -> CompliantDataFrame[Any, Any, Any]:
        result = self.native.compute(**kwargs)

        if backend is None or backend is Implementation.PANDAS:
            from narwhals._pandas_like.dataframe import PandasLikeDataFrame

            return PandasLikeDataFrame(
                result,
                implementation=Implementation.PANDAS,
                backend_version=parse_version(pd),
                version=self._version,
                validate_column_names=True,
            )

        if backend is Implementation.POLARS:
            import polars as pl  # ignore-banned-import

            from narwhals._polars.dataframe import PolarsDataFrame

            return PolarsDataFrame(
                pl.from_pandas(result),
                backend_version=parse_version(pl),
                version=self._version,
            )

        if backend is Implementation.PYARROW:
            import pyarrow as pa  # ignore-banned-import

            from narwhals._arrow.dataframe import ArrowDataFrame

            return ArrowDataFrame(
                pa.Table.from_pandas(result),
                backend_version=parse_version(pa),
                version=self._version,
                validate_column_names=True,
            )

        msg = f"Unsupported `backend` value: {backend}"  # pragma: no cover
        raise ValueError(msg)  # pragma: no cover

    @property
    def columns(self) -> list[str]:
        if self._cached_columns is None:
            self._cached_columns = (
                list(self.schema)
                if self._cached_schema is not None
                else self.native.columns.tolist()
            )
        return self._cached_columns

    def filter(self, predicate: DaskExpr) -> Self:
        # `[0]` is safe as the predicate's expression only returns a single column
        mask = predicate(self)[0]
        return self._with_native(self.native.loc[mask])

    def simple_select(self, *column_names: str) -> Self:
        native = select_columns_by_name(
            self.native, list(column_names), self._backend_version, self._implementation
        )
        return self._with_native(native)

    def aggregate(self, *exprs: DaskExpr) -> Self:
        new_series = evaluate_exprs(self, *exprs)
        df = dd.concat([val.rename(name) for name, val in new_series], axis=1)
        return self._with_native(df)

    def select(self, *exprs: DaskExpr) -> Self:
        new_series = evaluate_exprs(self, *exprs)
        df = select_columns_by_name(
            self.native.assign(**dict(new_series)),
            [s[0] for s in new_series],
            self._backend_version,
            self._implementation,
        )
        return self._with_native(df)

    def drop_nulls(self, subset: Sequence[str] | None) -> Self:
        if subset is None:
            return self._with_native(self.native.dropna())
        plx = self.__narwhals_namespace__()
        return self.filter(~plx.any_horizontal(plx.col(*subset).is_null()))

    @property
    def schema(self) -> dict[str, DType]:
        if self._cached_schema is None:
            native_dtypes = self.native.dtypes
            self._cached_schema = {
                col: native_to_narwhals_dtype(
                    native_dtypes[col], self._version, self._implementation
                )
                for col in self.native.columns
            }
        return self._cached_schema

    def collect_schema(self) -> dict[str, DType]:
        return self.schema

    def drop(self, columns: Sequence[str], *, strict: bool) -> Self:
        to_drop = parse_columns_to_drop(
            compliant_frame=self, columns=columns, strict=strict
        )

        return self._with_native(self.native.drop(columns=to_drop))

    def with_row_index(self, name: str) -> Self:
        # Implementation is based on the following StackOverflow reply:
        # https://stackoverflow.com/questions/60831518/in-dask-how-does-one-add-a-range-of-integersauto-increment-to-a-new-column/60852409#60852409
        return self._with_native(
            add_row_index(self.native, name, self._backend_version, self._implementation)
        )

    def rename(self, mapping: Mapping[str, str]) -> Self:
        return self._with_native(self.native.rename(columns=mapping))

    def head(self, n: int) -> Self:
        return self._with_native(self.native.head(n=n, compute=False, npartitions=-1))

    def unique(
        self, subset: Sequence[str] | None, *, keep: LazyUniqueKeepStrategy
    ) -> Self:
        check_column_exists(self.columns, subset)
        if keep == "none":
            subset = subset or self.columns
            token = generate_temporary_column_name(n_bytes=8, columns=subset)
            ser = self.native.groupby(subset).size().rename(token)
            ser = ser[ser == 1]
            unique = ser.reset_index().drop(columns=token)
            result = self.native.merge(unique, on=subset, how="inner")
        else:
            mapped_keep = {"any": "first"}.get(keep, keep)
            result = self.native.drop_duplicates(subset=subset, keep=mapped_keep)
        return self._with_native(result)

    def sort(
        self,
        *by: str,
        descending: bool | Sequence[bool],
        nulls_last: bool,
    ) -> Self:
        if isinstance(descending, bool):
            ascending: bool | list[bool] = not descending
        else:
            ascending = [not d for d in descending]
        position = "last" if nulls_last else "first"
        return self._with_native(
            self.native.sort_values(list(by), ascending=ascending, na_position=position)
        )

    def join(
        self,
        other: Self,
        *,
        how: JoinStrategy,
        left_on: Sequence[str] | None,
        right_on: Sequence[str] | None,
        suffix: str,
    ) -> Self:
        if how == "cross":
            key_token = generate_temporary_column_name(
                n_bytes=8, columns=[*self.columns, *other.columns]
            )

            return self._with_native(
                self.native.assign(**{key_token: 0})
                .merge(
                    other.native.assign(**{key_token: 0}),
                    how="inner",
                    left_on=key_token,
                    right_on=key_token,
                    suffixes=("", suffix),
                )
                .drop(columns=key_token)
            )

        if how == "anti":
            indicator_token = generate_temporary_column_name(
                n_bytes=8, columns=[*self.columns, *other.columns]
            )

            if right_on is None:  # pragma: no cover
                msg = "`right_on` cannot be `None` in anti-join"
                raise TypeError(msg)
            other_native = (
                select_columns_by_name(
                    other.native,
                    list(right_on),
                    self._backend_version,
                    self._implementation,
                )
                .rename(  # rename to avoid creating extra columns in join
                    columns=dict(zip(right_on, left_on))  # type: ignore[arg-type]
                )
                .drop_duplicates()
            )
            df = self.native.merge(
                other_native,
                how="outer",
                indicator=indicator_token,  # pyright: ignore[reportArgumentType]
                left_on=left_on,
                right_on=left_on,
            )
            return self._with_native(
                df[df[indicator_token] == "left_only"].drop(columns=[indicator_token])
            )

        if how == "semi":
            if right_on is None:  # pragma: no cover
                msg = "`right_on` cannot be `None` in semi-join"
                raise TypeError(msg)
            other_native = (
                select_columns_by_name(
                    other.native,
                    list(right_on),
                    self._backend_version,
                    self._implementation,
                )
                .rename(  # rename to avoid creating extra columns in join
                    columns=dict(zip(right_on, left_on))  # type: ignore[arg-type]
                )
                .drop_duplicates()  # avoids potential rows duplication from inner join
            )
            return self._with_native(
                self.native.merge(
                    other_native, how="inner", left_on=left_on, right_on=left_on
                )
            )

        if how == "left":
            result_native = self.native.merge(
                other.native,
                how="left",
                left_on=left_on,
                right_on=right_on,
                suffixes=("", suffix),
            )
            extra = []
            for left_key, right_key in zip(left_on, right_on):  # type: ignore[arg-type]
                if right_key != left_key and right_key not in self.columns:
                    extra.append(right_key)
                elif right_key != left_key:
                    extra.append(f"{right_key}_right")
            return self._with_native(result_native.drop(columns=extra))

        if how == "full":
            # dask does not retain keys post-join
            # we must append the suffix to each key before-hand

            # help mypy
            assert left_on is not None  # noqa: S101
            assert right_on is not None  # noqa: S101

            right_on_mapper = _remap_full_join_keys(left_on, right_on, suffix)
            other_native = other.native.rename(columns=right_on_mapper)
            check_column_names_are_unique(other_native.columns)
            right_on = list(right_on_mapper.values())  # we now have the suffixed keys
            return self._with_native(
                self.native.merge(
                    other_native,
                    left_on=left_on,
                    right_on=right_on,
                    how="outer",
                    suffixes=("", suffix),
                )
            )

        return self._with_native(
            self.native.merge(
                other.native,
                left_on=left_on,
                right_on=right_on,
                how=how,
                suffixes=("", suffix),
            )
        )

    def join_asof(
        self,
        other: Self,
        *,
        left_on: str | None,
        right_on: str | None,
        by_left: Sequence[str] | None,
        by_right: Sequence[str] | None,
        strategy: AsofJoinStrategy,
        suffix: str,
    ) -> Self:
        plx = self.__native_namespace__()
        return self._with_native(
            plx.merge_asof(
                self.native,
                other.native,
                left_on=left_on,
                right_on=right_on,
                left_by=by_left,
                right_by=by_right,
                direction=strategy,
                suffixes=("", suffix),
            ),
        )

    def group_by(self, *by: str, drop_null_keys: bool) -> DaskLazyGroupBy:
        from narwhals._dask.group_by import DaskLazyGroupBy

        return DaskLazyGroupBy(self, by, drop_null_keys=drop_null_keys)

    def tail(self, n: int) -> Self:  # pragma: no cover
        native_frame = self.native
        n_partitions = native_frame.npartitions

        if n_partitions == 1:
            return self._with_native(self.native.tail(n=n, compute=False))
        else:
            msg = "`LazyFrame.tail` is not supported for Dask backend with multiple partitions."
            raise NotImplementedError(msg)

    def gather_every(self, n: int, offset: int) -> Self:
        row_index_token = generate_temporary_column_name(n_bytes=8, columns=self.columns)
        plx = self.__narwhals_namespace__()
        return (
            self.with_row_index(row_index_token)
            .filter(
                (plx.col(row_index_token) >= offset)
                & ((plx.col(row_index_token) - offset) % n == 0)
            )
            .drop([row_index_token], strict=False)
        )

    def unpivot(
        self,
        on: Sequence[str] | None,
        index: Sequence[str] | None,
        variable_name: str,
        value_name: str,
    ) -> Self:
        return self._with_native(
            self.native.melt(
                id_vars=index,
                value_vars=on,
                var_name=variable_name,
                value_name=value_name,
            )
        )

    explode = not_implemented()
