from __future__ import annotations

from typing import TYPE_CHECKING
from typing import Any
from typing import Literal
from typing import Sequence

from narwhals._polars.namespace import PolarsNamespace
from narwhals._polars.utils import convert_str_slice_to_int_slice
from narwhals._polars.utils import extract_args_kwargs
from narwhals._polars.utils import native_to_narwhals_dtype
from narwhals.exceptions import ColumnNotFoundError
from narwhals.exceptions import InvalidIntoExprError
from narwhals.utils import Implementation
from narwhals.utils import is_sequence_but_not_str
from narwhals.utils import parse_columns_to_drop

if TYPE_CHECKING:
    from types import ModuleType

    import numpy as np
    import polars as pl
    from typing_extensions import Self

    from narwhals.dtypes import DType
    from narwhals.typing import DTypes


class PolarsDataFrame:
    def __init__(
        self, df: pl.DataFrame, *, backend_version: tuple[int, ...], dtypes: DTypes
    ) -> None:
        self._native_frame = df
        self._backend_version = backend_version
        self._implementation = Implementation.POLARS
        self._dtypes = dtypes

    def __repr__(self) -> str:  # pragma: no cover
        return "PolarsDataFrame"

    def __narwhals_dataframe__(self) -> Self:
        return self

    def __narwhals_namespace__(self) -> PolarsNamespace:
        return PolarsNamespace(backend_version=self._backend_version, dtypes=self._dtypes)

    def __native_namespace__(self: Self) -> ModuleType:
        if self._implementation is Implementation.POLARS:
            return self._implementation.to_native_namespace()

        msg = f"Expected polars, got: {type(self._implementation)}"  # pragma: no cover
        raise AssertionError(msg)

    def _change_dtypes(self, dtypes: DTypes) -> Self:
        return self.__class__(
            self._native_frame, backend_version=self._backend_version, dtypes=dtypes
        )

    def _from_native_frame(self, df: Any) -> Self:
        return self.__class__(
            df, backend_version=self._backend_version, dtypes=self._dtypes
        )

    def _from_native_object(self, obj: Any) -> Any:
        import polars as pl  # ignore-banned-import()

        if isinstance(obj, pl.Series):
            from narwhals._polars.series import PolarsSeries

            return PolarsSeries(
                obj, backend_version=self._backend_version, dtypes=self._dtypes
            )
        if isinstance(obj, pl.DataFrame):
            return self._from_native_frame(obj)
        # scalar
        return obj

    def __getattr__(self, attr: str) -> Any:
        if attr == "collect":  # pragma: no cover
            raise AttributeError
        if attr == "schema":
            schema = self._native_frame.schema
            return {
                name: native_to_narwhals_dtype(dtype, self._dtypes, self._backend_version)
                for name, dtype in schema.items()
            }

        def func(*args: Any, **kwargs: Any) -> Any:
            import polars as pl  # ignore-banned-import()

            args, kwargs = extract_args_kwargs(args, kwargs)  # type: ignore[assignment]
            try:
                return self._from_native_object(
                    getattr(self._native_frame, attr)(*args, **kwargs)
                )
            except pl.exceptions.ColumnNotFoundError as e:
                msg = f"{e!s}\n\nHint: Did you mean one of these columns: {self.columns}?"
                raise ColumnNotFoundError(msg) from e
            except TypeError as e:
                e_str = str(e)
                if (
                    "cannot create expression literal" in e_str
                    or "invalid literal" in e_str
                ):
                    raise InvalidIntoExprError(e_str) from e
                raise

        return func

    def __array__(self, dtype: Any | None = None, copy: bool | None = None) -> np.ndarray:
        if self._backend_version < (0, 20, 28) and copy is not None:
            msg = "`copy` in `__array__` is only supported for Polars>=0.20.28"
            raise NotImplementedError(msg)
        if self._backend_version < (0, 20, 28):
            return self._native_frame.__array__(dtype)
        return self._native_frame.__array__(dtype)

    def collect_schema(self) -> dict[str, DType]:
        if self._backend_version < (1,):
            return {
                name: native_to_narwhals_dtype(dtype, self._dtypes, self._backend_version)
                for name, dtype in self._native_frame.schema.items()
            }
        else:
            return {
                name: native_to_narwhals_dtype(dtype, self._dtypes, self._backend_version)
                for name, dtype in self._native_frame.collect_schema().items()
            }

    @property
    def shape(self) -> tuple[int, int]:
        return self._native_frame.shape

    def __getitem__(self, item: Any) -> Any:
        if self._backend_version > (0, 20, 30):
            return self._from_native_object(self._native_frame.__getitem__(item))
        else:  # pragma: no cover
            # TODO(marco): we can delete this branch after Polars==0.20.30 becomes the minimum
            # Polars version we support
            if isinstance(item, tuple):
                item = tuple(list(i) if is_sequence_but_not_str(i) else i for i in item)

            columns = self.columns
            if isinstance(item, tuple) and len(item) == 2 and isinstance(item[1], slice):
                if item[1] == slice(None):
                    if isinstance(item[0], Sequence) and not len(item[0]):
                        return self._from_native_frame(self._native_frame[0:0])
                    return self._from_native_frame(
                        self._native_frame.__getitem__(item[0])
                    )
                if isinstance(item[1].start, str) or isinstance(item[1].stop, str):
                    start, stop, step = convert_str_slice_to_int_slice(item[1], columns)
                    return self._from_native_frame(
                        self._native_frame.select(columns[start:stop:step]).__getitem__(
                            item[0]
                        )
                    )
                if isinstance(item[1].start, int) or isinstance(item[1].stop, int):
                    return self._from_native_frame(
                        self._native_frame.select(
                            columns[item[1].start : item[1].stop : item[1].step]
                        ).__getitem__(item[0])
                    )
                msg = f"Expected slice of integers or strings, got: {type(item[1])}"  # pragma: no cover
                raise TypeError(msg)  # pragma: no cover
            import polars as pl  # ignore-banned-import()

            if (
                isinstance(item, tuple)
                and (len(item) == 2)
                and is_sequence_but_not_str(item[1])
                and (len(item[1]) == 0)
            ):
                result = self._native_frame.select(item[1])
            elif isinstance(item, slice) and (
                isinstance(item.start, str) or isinstance(item.stop, str)
            ):
                start, stop, step = convert_str_slice_to_int_slice(item, columns)
                return self._from_native_frame(
                    self._native_frame.select(columns[start:stop:step])
                )
            elif is_sequence_but_not_str(item) and (len(item) == 0):
                result = self._native_frame.slice(0, 0)
            else:
                result = self._native_frame.__getitem__(item)
            if isinstance(result, pl.Series):
                from narwhals._polars.series import PolarsSeries

                return PolarsSeries(
                    result, backend_version=self._backend_version, dtypes=self._dtypes
                )
            return self._from_native_object(result)

    def get_column(self, name: str) -> Any:
        from narwhals._polars.series import PolarsSeries

        return PolarsSeries(
            self._native_frame.get_column(name),
            backend_version=self._backend_version,
            dtypes=self._dtypes,
        )

    def is_empty(self) -> bool:
        return len(self._native_frame) == 0

    @property
    def columns(self) -> list[str]:
        return self._native_frame.columns

    def lazy(self) -> PolarsLazyFrame:
        return PolarsLazyFrame(
            self._native_frame.lazy(),
            backend_version=self._backend_version,
            dtypes=self._dtypes,
        )

    def to_dict(self, *, as_series: bool) -> Any:
        df = self._native_frame

        if as_series:
            from narwhals._polars.series import PolarsSeries

            return {
                name: PolarsSeries(
                    col, backend_version=self._backend_version, dtypes=self._dtypes
                )
                for name, col in df.to_dict(as_series=True).items()
            }
        else:
            return df.to_dict(as_series=False)

    def group_by(self, *by: str, drop_null_keys: bool) -> Any:
        from narwhals._polars.group_by import PolarsGroupBy

        return PolarsGroupBy(self, list(by), drop_null_keys=drop_null_keys)

    def with_row_index(self, name: str) -> Any:
        if self._backend_version < (0, 20, 4):
            return self._from_native_frame(self._native_frame.with_row_count(name))
        return self._from_native_frame(self._native_frame.with_row_index(name))

    def drop(self: Self, columns: list[str], strict: bool) -> Self:  # noqa: FBT001
        to_drop = parse_columns_to_drop(
            compliant_frame=self, columns=columns, strict=strict
        )
        return self._from_native_frame(self._native_frame.drop(to_drop))

    def unpivot(
        self: Self,
        on: str | list[str] | None,
        index: str | list[str] | None,
        variable_name: str | None,
        value_name: str | None,
    ) -> Self:
        if self._backend_version < (1, 0, 0):
            return self._from_native_frame(
                self._native_frame.melt(
                    id_vars=index,
                    value_vars=on,
                    variable_name=variable_name,
                    value_name=value_name,
                )
            )
        return self._from_native_frame(
            self._native_frame.unpivot(
                on=on, index=index, variable_name=variable_name, value_name=value_name
            )
        )

    def pivot(
        self: Self,
        on: str | list[str],
        *,
        index: str | list[str] | None = None,
        values: str | list[str] | None = None,
        aggregate_function: Literal[
            "min", "max", "first", "last", "sum", "mean", "median", "len"
        ]
        | None = None,
        maintain_order: bool = True,
        sort_columns: bool = False,
        separator: str = "_",
    ) -> Self:
        if self._backend_version < (1, 0, 0):  # pragma: no cover
            msg = "`pivot` is only supported for Polars>=1.0.0"
            raise NotImplementedError(msg)
        result = self._native_frame.pivot(
            on,
            index=index,
            values=values,
            aggregate_function=aggregate_function,
            maintain_order=maintain_order,
            sort_columns=sort_columns,
            separator=separator,
        )
        return self._from_native_object(result)  # type: ignore[no-any-return]


class PolarsLazyFrame:
    def __init__(
        self, df: pl.LazyFrame, *, backend_version: tuple[int, ...], dtypes: DTypes
    ) -> None:
        self._native_frame = df
        self._backend_version = backend_version
        self._implementation = Implementation.POLARS
        self._dtypes = dtypes

    def __repr__(self) -> str:  # pragma: no cover
        return "PolarsLazyFrame"

    def __narwhals_lazyframe__(self) -> Self:
        return self

    def __narwhals_namespace__(self) -> PolarsNamespace:
        return PolarsNamespace(backend_version=self._backend_version, dtypes=self._dtypes)

    def __native_namespace__(self: Self) -> ModuleType:
        if self._implementation is Implementation.POLARS:
            return self._implementation.to_native_namespace()

        msg = f"Expected polars, got: {type(self._implementation)}"  # pragma: no cover
        raise AssertionError(msg)

    def _from_native_frame(self, df: Any) -> Self:
        return self.__class__(
            df, backend_version=self._backend_version, dtypes=self._dtypes
        )

    def _change_dtypes(self, dtypes: DTypes) -> Self:
        return self.__class__(
            self._native_frame, backend_version=self._backend_version, dtypes=dtypes
        )

    def __getattr__(self, attr: str) -> Any:
        def func(*args: Any, **kwargs: Any) -> Any:
            import polars as pl  # ignore-banned-import

            args, kwargs = extract_args_kwargs(args, kwargs)  # type: ignore[assignment]
            try:
                return self._from_native_frame(
                    getattr(self._native_frame, attr)(*args, **kwargs)
                )
            except pl.exceptions.ColumnNotFoundError as e:  # pragma: no cover
                raise ColumnNotFoundError(str(e)) from e
            except TypeError as e:
                e_str = str(e)
                if (
                    "cannot create expression literal" in e_str
                    or "invalid literal" in e_str
                ):
                    raise InvalidIntoExprError(e_str) from e
                raise

        return func

    @property
    def columns(self) -> list[str]:
        return self._native_frame.columns

    @property
    def schema(self) -> dict[str, Any]:
        schema = self._native_frame.schema
        return {
            name: native_to_narwhals_dtype(dtype, self._dtypes, self._backend_version)
            for name, dtype in schema.items()
        }

    def collect_schema(self) -> dict[str, DType]:
        if self._backend_version < (1,):
            return {
                name: native_to_narwhals_dtype(dtype, self._dtypes, self._backend_version)
                for name, dtype in self._native_frame.schema.items()
            }
        else:
            return {
                name: native_to_narwhals_dtype(dtype, self._dtypes, self._backend_version)
                for name, dtype in self._native_frame.collect_schema().items()
            }

    def collect(self) -> PolarsDataFrame:
        import polars as pl  # ignore-banned-import

        try:
            result = self._native_frame.collect()
        except pl.exceptions.ColumnNotFoundError as e:
            raise ColumnNotFoundError(str(e)) from e

        return PolarsDataFrame(
            result,
            backend_version=self._backend_version,
            dtypes=self._dtypes,
        )

    def group_by(self, *by: str, drop_null_keys: bool) -> Any:
        from narwhals._polars.group_by import PolarsLazyGroupBy

        return PolarsLazyGroupBy(self, list(by), drop_null_keys=drop_null_keys)

    def with_row_index(self, name: str) -> Any:
        if self._backend_version < (0, 20, 4):
            return self._from_native_frame(self._native_frame.with_row_count(name))
        return self._from_native_frame(self._native_frame.with_row_index(name))

    def drop(self: Self, columns: list[str], strict: bool) -> Self:  # noqa: FBT001
        if self._backend_version < (1, 0, 0):
            return self._from_native_frame(self._native_frame.drop(columns))
        return self._from_native_frame(self._native_frame.drop(columns, strict=strict))

    def unpivot(
        self: Self,
        on: str | list[str] | None,
        index: str | list[str] | None,
        variable_name: str | None,
        value_name: str | None,
    ) -> Self:
        if self._backend_version < (1, 0, 0):
            return self._from_native_frame(
                self._native_frame.melt(
                    id_vars=index,
                    value_vars=on,
                    variable_name=variable_name,
                    value_name=value_name,
                )
            )
        return self._from_native_frame(
            self._native_frame.unpivot(
                on=on, index=index, variable_name=variable_name, value_name=value_name
            )
        )
