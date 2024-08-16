from __future__ import annotations

from typing import TYPE_CHECKING
from typing import Any

from narwhals._polars.namespace import PolarsNamespace
from narwhals._polars.utils import extract_args_kwargs
from narwhals._polars.utils import translate_dtype
from narwhals.dependencies import get_polars
from narwhals.utils import Implementation
from narwhals.utils import parse_columns_to_drop

if TYPE_CHECKING:
    from typing_extensions import Self


class PolarsDataFrame:
    def __init__(self, df: Any, *, backend_version: tuple[int, ...]) -> None:
        self._native_frame = df
        self._implementation = Implementation.POLARS
        self._backend_version = backend_version

    def __repr__(self) -> str:  # pragma: no cover
        return "PolarsDataFrame"

    def __narwhals_dataframe__(self) -> Self:
        return self

    def __narwhals_namespace__(self) -> PolarsNamespace:
        return PolarsNamespace(backend_version=self._backend_version)

    def __native_namespace__(self) -> Any:
        return get_polars()

    def _from_native_frame(self, df: Any) -> Self:
        return self.__class__(df, backend_version=self._backend_version)

    def _from_native_object(self, obj: Any) -> Any:
        pl = get_polars()
        if isinstance(obj, pl.Series):
            from narwhals._polars.series import PolarsSeries

            return PolarsSeries(obj, backend_version=self._backend_version)
        if isinstance(obj, pl.DataFrame):
            return self._from_native_frame(obj)
        # scalar
        return obj

    def __getattr__(self, attr: str) -> Any:
        if attr == "collect":  # pragma: no cover
            raise AttributeError

        def func(*args: Any, **kwargs: Any) -> Any:
            args, kwargs = extract_args_kwargs(args, kwargs)  # type: ignore[assignment]
            return self._from_native_object(
                getattr(self._native_frame, attr)(*args, **kwargs)
            )

        return func

    @property
    def schema(self) -> dict[str, Any]:
        schema = self._native_frame.schema
        return {name: translate_dtype(dtype) for name, dtype in schema.items()}

    def collect_schema(self) -> dict[str, Any]:
        if self._backend_version < (1,):  # pragma: no cover
            schema = self._native_frame.schema
        else:
            schema = dict(self._native_frame.collect_schema())
        return {name: translate_dtype(dtype) for name, dtype in schema.items()}

    @property
    def shape(self) -> tuple[int, int]:
        return self._native_frame.shape  # type: ignore[no-any-return]

    def __getitem__(self, item: Any) -> Any:
        pl = get_polars()
        result = self._native_frame.__getitem__(item)
        if isinstance(result, pl.Series):
            from narwhals._polars.series import PolarsSeries

            return PolarsSeries(result, backend_version=self._backend_version)
        return self._from_native_object(result)

    def get_column(self, name: str) -> Any:
        from narwhals._polars.series import PolarsSeries

        return PolarsSeries(
            self._native_frame.get_column(name), backend_version=self._backend_version
        )

    def is_empty(self) -> bool:
        return len(self._native_frame) == 0

    @property
    def columns(self) -> list[str]:
        return self._native_frame.columns  # type: ignore[no-any-return]

    def lazy(self) -> PolarsLazyFrame:
        return PolarsLazyFrame(
            self._native_frame.lazy(), backend_version=self._backend_version
        )

    def to_dict(self, *, as_series: bool) -> Any:
        df = self._native_frame

        if as_series:
            from narwhals._polars.series import PolarsSeries

            return {
                name: PolarsSeries(col, backend_version=self._backend_version)
                for name, col in df.to_dict(as_series=True).items()
            }
        else:
            return df.to_dict(as_series=False)

    def group_by(self, *by: str) -> Any:
        from narwhals._polars.group_by import PolarsGroupBy

        return PolarsGroupBy(self, list(by))

    def with_row_index(self, name: str) -> Any:
        if self._backend_version < (0, 20, 4):  # pragma: no cover
            return self._from_native_frame(self._native_frame.with_row_count(name))
        return self._from_native_frame(self._native_frame.with_row_index(name))

    def drop(self: Self, columns: list[str], strict: bool) -> Self:  # noqa: FBT001
        if self._backend_version < (1, 0, 0):  # pragma: no cover
            to_drop = parse_columns_to_drop(
                compliant_frame=self, columns=columns, strict=strict
            )
            return self._from_native_frame(self._native_frame.drop(to_drop))
        return self._from_native_frame(self._native_frame.drop(columns, strict=strict))


class PolarsLazyFrame:
    def __init__(self, df: Any, *, backend_version: tuple[int, ...]) -> None:
        self._native_frame = df
        self._backend_version = backend_version

    def __repr__(self) -> str:  # pragma: no cover
        return "PolarsLazyFrame"

    def __narwhals_lazyframe__(self) -> Self:
        return self

    def __narwhals_namespace__(self) -> PolarsNamespace:
        return PolarsNamespace(backend_version=self._backend_version)

    def __native_namespace__(self) -> Any:  # pragma: no cover
        return get_polars()

    def _from_native_frame(self, df: Any) -> Self:
        return self.__class__(df, backend_version=self._backend_version)

    def __getattr__(self, attr: str) -> Any:
        def func(*args: Any, **kwargs: Any) -> Any:
            args, kwargs = extract_args_kwargs(args, kwargs)  # type: ignore[assignment]
            return self._from_native_frame(
                getattr(self._native_frame, attr)(*args, **kwargs)
            )

        return func

    @property
    def columns(self) -> list[str]:
        return self._native_frame.columns  # type: ignore[no-any-return]

    @property
    def schema(self) -> dict[str, Any]:
        schema = self._native_frame.schema
        return {name: translate_dtype(dtype) for name, dtype in schema.items()}

    def collect_schema(self) -> dict[str, Any]:
        if self._backend_version < (1,):  # pragma: no cover
            schema = self._native_frame.schema
        else:
            schema = dict(self._native_frame.collect_schema())
        return {name: translate_dtype(dtype) for name, dtype in schema.items()}

    def collect(self) -> PolarsDataFrame:
        return PolarsDataFrame(
            self._native_frame.collect(), backend_version=self._backend_version
        )

    def group_by(self, *by: str) -> Any:
        from narwhals._polars.group_by import PolarsLazyGroupBy

        return PolarsLazyGroupBy(self, list(by))

    def with_row_index(self, name: str) -> Any:
        if self._backend_version < (0, 20, 4):  # pragma: no cover
            return self._from_native_frame(self._native_frame.with_row_count(name))
        return self._from_native_frame(self._native_frame.with_row_index(name))

    def drop(self: Self, columns: list[str], strict: bool) -> Self:  # noqa: FBT001
        if self._backend_version < (1, 0, 0):  # pragma: no cover
            return self._from_native_frame(self._native_frame.drop(columns))
        return self._from_native_frame(self._native_frame.drop(columns, strict=strict))
