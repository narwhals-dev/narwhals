from __future__ import annotations

from typing import TYPE_CHECKING
from typing import Any
from typing import Literal

from narwhals._polars.namespace import PolarsNamespace
from narwhals._polars.utils import extract_args_kwargs
from narwhals._polars.utils import translate_dtype
from narwhals.dependencies import get_polars

if TYPE_CHECKING:
    from typing_extensions import Self


class PolarsDataFrame:
    def __init__(self, df: Any, *, backend_version: tuple[int, ...]) -> None:
        self._native_dataframe = df
        self._backend_version = backend_version

    def __repr__(self) -> str:  # pragma: no cover
        return "PolarsDataFrame"

    def __narwhals_dataframe__(self) -> Self:
        return self

    def __narwhals_namespace__(self) -> PolarsNamespace:
        return PolarsNamespace(backend_version=self._backend_version)

    def __native_namespace__(self) -> Any:
        return get_polars()

    def _from_native_dataframe(self, df: Any) -> Self:
        return self.__class__(df, backend_version=self._backend_version)

    def _from_native_object(self, obj: Any) -> Any:
        pl = get_polars()
        if isinstance(obj, pl.Series):
            from narwhals._polars.series import PolarsSeries

            return PolarsSeries(obj, backend_version=self._backend_version)
        if isinstance(obj, pl.DataFrame):
            return self._from_native_dataframe(obj)
        # scalar
        return obj

    def __getattr__(self, attr: str) -> Any:
        if attr == "collect":  # pragma: no cover
            raise AttributeError

        def func(*args: Any, **kwargs: Any) -> Any:
            args, kwargs = extract_args_kwargs(args, kwargs)  # type: ignore[assignment]
            return self._from_native_object(
                getattr(self._native_dataframe, attr)(*args, **kwargs)
            )

        return func

    @property
    def schema(self) -> dict[str, Any]:
        schema = self._native_dataframe.schema
        return {name: translate_dtype(dtype) for name, dtype in schema.items()}

    def collect_schema(self) -> dict[str, Any]:
        if self._backend_version < (1,):  # pragma: no cover
            schema = self._native_dataframe.schema
        else:
            schema = dict(self._native_dataframe.collect_schema())
        return {name: translate_dtype(dtype) for name, dtype in schema.items()}

    @property
    def shape(self) -> tuple[int, int]:
        return self._native_dataframe.shape  # type: ignore[no-any-return]

    def __getitem__(self, item: Any) -> Any:
        pl = get_polars()
        result = self._native_dataframe.__getitem__(item)
        if isinstance(result, pl.Series):
            from narwhals._polars.series import PolarsSeries

            return PolarsSeries(result, backend_version=self._backend_version)
        return self._from_native_object(result)

    def get_column(self, name: str) -> Any:
        from narwhals._polars.series import PolarsSeries

        return PolarsSeries(
            self._native_dataframe.get_column(name), backend_version=self._backend_version
        )

    def is_empty(self) -> bool:
        return len(self._native_dataframe) == 0

    @property
    def columns(self) -> list[str]:
        return self._native_dataframe.columns  # type: ignore[no-any-return]

    def lazy(self) -> PolarsLazyFrame:
        return PolarsLazyFrame(
            self._native_dataframe.lazy(), backend_version=self._backend_version
        )

    def to_dict(self, *, as_series: bool) -> Any:
        df = self._native_dataframe

        if as_series:
            from narwhals._polars.series import PolarsSeries

            return {
                name: PolarsSeries(col, backend_version=self._backend_version)
                for name, col in df.to_dict(as_series=True).items()
            }
        else:
            return df.to_dict(as_series=False)

    def group_by(self, by: list[str]) -> Any:
        from narwhals._polars.group_by import PolarsGroupBy

        return PolarsGroupBy(self, by)

    def with_row_index(self, name: str) -> Any:
        if self._backend_version < (0, 20, 4):  # pragma: no cover
            return self._from_native_dataframe(
                self._native_dataframe.with_row_count(name)
            )
        return self._from_native_dataframe(self._native_dataframe.with_row_index(name))

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
        if isinstance(on, str):
            on = [on]
        if isinstance(index, str):
            index = [index]

        if values is None:
            values_ = [c for c in self.columns if c not in {*on, *index}]  # type: ignore[misc]
        elif isinstance(values, str):  # pragma: no cover
            values_ = [values]
        else:
            values_ = values

        kwargs = {
            "index": index,
            "values": values_,
            "aggregate_function": aggregate_function,
            "maintain_order": maintain_order,
            "sort_columns": sort_columns,
            "separator": separator,
        }

        if self._backend_version < (1, 0, 0):  # pragma: no cover
            import re

            if self._backend_version < (0, 20, 5) and aggregate_function == "len":
                kwargs["aggregate_function"] = "count"

            result = self._native_dataframe.pivot(columns=on, **kwargs)
            cols = result.columns
            combined_on = r"|".join(on)
            result = result.rename(
                {
                    c: re.sub(
                        rf"{separator}{combined_on}{separator}", separator, c, count=1
                    )
                    for c in cols
                }
            )
        else:
            result = self._native_dataframe.pivot(on=on, **kwargs)

        return self._from_native_dataframe(result)


class PolarsLazyFrame:
    def __init__(self, df: Any, *, backend_version: tuple[int, ...]) -> None:
        self._native_dataframe = df
        self._backend_version = backend_version

    def __repr__(self) -> str:  # pragma: no cover
        return "PolarsLazyFrame"

    def __narwhals_lazyframe__(self) -> Self:
        return self

    def __narwhals_namespace__(self) -> PolarsNamespace:
        return PolarsNamespace(backend_version=self._backend_version)

    def __native_namespace__(self) -> Any:  # pragma: no cover
        return get_polars()

    def _from_native_dataframe(self, df: Any) -> Self:
        return self.__class__(df, backend_version=self._backend_version)

    def __getattr__(self, attr: str) -> Any:
        def func(*args: Any, **kwargs: Any) -> Any:
            args, kwargs = extract_args_kwargs(args, kwargs)  # type: ignore[assignment]
            return self._from_native_dataframe(
                getattr(self._native_dataframe, attr)(*args, **kwargs)
            )

        return func

    @property
    def columns(self) -> list[str]:
        return self._native_dataframe.columns  # type: ignore[no-any-return]

    @property
    def schema(self) -> dict[str, Any]:
        schema = self._native_dataframe.schema
        return {name: translate_dtype(dtype) for name, dtype in schema.items()}

    def collect_schema(self) -> dict[str, Any]:
        if self._backend_version < (1,):  # pragma: no cover
            schema = self._native_dataframe.schema
        else:
            schema = dict(self._native_dataframe.collect_schema())
        return {name: translate_dtype(dtype) for name, dtype in schema.items()}

    def collect(self) -> PolarsDataFrame:
        return PolarsDataFrame(
            self._native_dataframe.collect(), backend_version=self._backend_version
        )

    def group_by(self, by: list[str]) -> Any:
        from narwhals._polars.group_by import PolarsLazyGroupBy

        return PolarsLazyGroupBy(self, by)

    def with_row_index(self, name: str) -> Any:
        if self._backend_version < (0, 20, 4):  # pragma: no cover
            return self._from_native_dataframe(
                self._native_dataframe.with_row_count(name)
            )
        return self._from_native_dataframe(self._native_dataframe.with_row_index(name))
