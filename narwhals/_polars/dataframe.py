from __future__ import annotations

from typing import TYPE_CHECKING
from typing import Any

from narwhals._polars.namespace import PolarsNamespace
from narwhals._polars.utils import extract_args_kwargs
from narwhals._polars.utils import translate_dtype
from narwhals.dependencies import get_polars

if TYPE_CHECKING:
    from typing_extensions import Self


class PolarsDataFrame:
    def __init__(self, df: Any) -> None:
        self._native_dataframe = df

    def __repr__(self) -> str:
        return "PolarsDataFrame"

    def __narwhals_dataframe__(self) -> Self:
        return self

    def __narwhals_namespace__(self) -> PolarsNamespace:
        return PolarsNamespace()

    def __native_namespace__(self) -> Any:
        return get_polars()

    def _from_native_frame(self, df: Any) -> Self:
        return self.__class__(df)

    def __getattr__(self, attr: str) -> Any:
        def func(*args: Any, **kwargs: Any) -> Any:
            args, kwargs = extract_args_kwargs(args, kwargs)  # type: ignore[assignment]
            return self._from_native_frame(
                getattr(self._native_dataframe, attr)(*args, **kwargs)
            )

        return func

    @property
    def schema(self) -> dict[str, Any]:
        schema = self._native_dataframe.schema
        return {name: translate_dtype(dtype) for name, dtype in schema.items()}

    @property
    def shape(self) -> tuple[int, int]:
        return self._native_dataframe.shape  # type: ignore[no-any-return]

    def __getitem__(self, item: Any) -> Any:
        pl = get_polars()
        result = self._native_dataframe.__getitem__(item)
        if isinstance(result, pl.Series):
            from narwhals._polars.series import PolarsSeries

            return PolarsSeries(result)
        return self._from_native_frame(result)

    def get_column(self, name: str) -> Any:
        from narwhals._polars.series import PolarsSeries

        return PolarsSeries(self._native_dataframe.get_column(name))

    def is_duplicated(self) -> Any:
        from narwhals._polars.series import PolarsSeries

        return PolarsSeries(self._native_dataframe.is_duplicated())

    def is_empty(self) -> bool:
        return len(self._native_dataframe) == 0

    @property
    def columns(self) -> list[str]:
        return self._native_dataframe.columns  # type: ignore[no-any-return]

    def lazy(self) -> PolarsLazyFrame:
        return PolarsLazyFrame(self._native_dataframe.lazy())


class PolarsLazyFrame:
    def __init__(self, df: Any) -> None:
        self._native_dataframe = df

    def __repr__(self) -> str:
        return "PolarsLazyFrame"

    def __narwhals_lazyframe__(self) -> Self:
        return self

    def __narwhals_namespace__(self) -> PolarsNamespace:
        return PolarsNamespace()

    def _from_native_frame(self, df: Any) -> Self:
        return self.__class__(df)

    def __getattr__(self, attr: str) -> Any:
        def func(*args: Any, **kwargs: Any) -> Any:
            args, kwargs = extract_args_kwargs(args, kwargs)  # type: ignore[assignment]
            return self._from_native_frame(
                getattr(self._native_dataframe, attr)(*args, **kwargs)
            )

        return func

    @property
    def columns(self) -> list[str]:
        return self._native_dataframe.columns  # type: ignore[no-any-return]

    def collect(self) -> PolarsDataFrame:
        return PolarsDataFrame(self._native_dataframe.collect())
