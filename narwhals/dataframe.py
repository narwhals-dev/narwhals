from __future__ import annotations

from typing import TYPE_CHECKING
from typing import Any
from typing import Iterable
from typing import Literal
from typing import Sequence

from narwhals.dtypes import to_narwhals_dtype
from narwhals.pandas_like.dataframe import PandasDataFrame
from narwhals.translate import get_cudf
from narwhals.translate import get_modin
from narwhals.translate import get_pandas
from narwhals.translate import get_polars

if TYPE_CHECKING:
    from typing_extensions import Self

    from narwhals.dtypes import DType
    from narwhals.group_by import GroupBy
    from narwhals.series import Series
    from narwhals.typing import IntoExpr


class BaseFrame:
    _dataframe: Any
    _implementation: str

    def _from_dataframe(self, df: Any) -> Self:
        # construct, preserving properties
        return self.__class__(  # type: ignore[call-arg]
            df,
            implementation=self._implementation,
        )

    def _flatten_and_extract(self, *args: Any, **kwargs: Any) -> Any:
        from narwhals.utils import flatten

        args = [self._extract_native(v) for v in flatten(args)]  # type: ignore[assignment]
        kwargs = {k: self._extract_native(v) for k, v in kwargs.items()}
        return args, kwargs

    def _extract_native(self, arg: Any) -> Any:
        from narwhals.expression import Expr
        from narwhals.pandas_like.namespace import PandasNamespace
        from narwhals.series import Series

        if isinstance(arg, BaseFrame):
            return arg._dataframe
        if isinstance(arg, Series):
            return arg._series
        if isinstance(arg, Expr):
            if self._implementation == "polars":
                import polars as pl

                return arg._call(pl)
            plx = PandasNamespace(implementation=self._implementation)
            return arg._call(plx)
        return arg

    def __repr__(self) -> str:  # pragma: no cover
        header = " Narwhals DataFrame                              "
        length = len(header)
        return (
            "┌"
            + "─" * length
            + "┐\n"
            + f"|{header}|\n"
            + "| Use `narwhals.to_native()` to see native output |\n"
            + "└"
            + "─" * length
            + "┘\n"
        )

    @property
    def schema(self) -> dict[str, DType]:
        return {
            k: to_narwhals_dtype(v, self._implementation)
            for k, v in self._dataframe.schema.items()
        }

    @property
    def columns(self) -> list[str]:
        return self._dataframe.columns  # type: ignore[no-any-return]

    def with_columns(
        self, *exprs: IntoExpr | Iterable[IntoExpr], **named_exprs: IntoExpr
    ) -> Self:
        exprs, named_exprs = self._flatten_and_extract(*exprs, **named_exprs)
        return self._from_dataframe(
            self._dataframe.with_columns(*exprs, **named_exprs),
        )

    def select(
        self,
        *exprs: IntoExpr | Iterable[IntoExpr],
        **named_exprs: IntoExpr,
    ) -> Self:
        exprs, named_exprs = self._flatten_and_extract(*exprs, **named_exprs)
        return self._from_dataframe(
            self._dataframe.select(*exprs, **named_exprs),
        )

    def rename(self, mapping: dict[str, str]) -> Self:
        return self._from_dataframe(self._dataframe.rename(mapping))

    def head(self, n: int) -> Self:
        return self._from_dataframe(self._dataframe.head(n))

    def drop(self, *columns: str | Iterable[str]) -> Self:
        return self._from_dataframe(self._dataframe.drop(*columns))

    def unique(self, subset: str | list[str]) -> Self:
        return self._from_dataframe(self._dataframe.unique(subset=subset))

    def filter(self, *predicates: IntoExpr | Iterable[IntoExpr]) -> Self:
        predicates, _ = self._flatten_and_extract(*predicates)
        return self._from_dataframe(
            self._dataframe.filter(*predicates),
        )

    def group_by(self, *keys: str | Iterable[str]) -> GroupBy:
        from narwhals.group_by import GroupBy

        # todo: groupby and lazygroupby
        return GroupBy(self, *keys)  # type: ignore[arg-type]

    def sort(
        self,
        by: str | Iterable[str],
        *more_by: str,
        descending: bool | Sequence[bool] = False,
    ) -> Self:
        return self._from_dataframe(
            self._dataframe.sort(by, *more_by, descending=descending)
        )

    def join(
        self,
        other: Self,
        *,
        how: Literal["inner"] = "inner",
        left_on: str | list[str],
        right_on: str | list[str],
    ) -> Self:
        return self._from_dataframe(
            self._dataframe.join(
                self._extract_native(other),
                how=how,
                left_on=left_on,
                right_on=right_on,
            )
        )


class DataFrame(BaseFrame):
    def __init__(
        self,
        df: Any,
        *,
        implementation: str | None = None,
    ) -> None:
        if implementation is not None:
            self._dataframe: Any = df
            self._implementation = implementation
            return
        if (pl := get_polars()) is not None and isinstance(df, pl.DataFrame):
            self._dataframe = df
            self._implementation = "polars"
        elif (pl := get_polars()) is not None and isinstance(df, pl.LazyFrame):
            raise TypeError(
                "Can't instantiate DataFrame from Polars LazyFrame. Call `collect()` first, or use `narwhals.LazyFrame` if you don't specifically require eager execution."
            )
        elif (pd := get_pandas()) is not None and isinstance(df, pd.DataFrame):
            self._dataframe = PandasDataFrame(df, implementation="pandas")
            self._implementation = "pandas"
        elif (mpd := get_modin()) is not None and isinstance(
            df, mpd.DataFrame
        ):  # pragma: no cover
            self._dataframe = PandasDataFrame(df, implementation="modin")
            self._implementation = "modin"
        elif (cudf := get_cudf()) is not None and isinstance(
            df, cudf.DataFrame
        ):  # pragma: no cover
            self._dataframe = PandasDataFrame(df, implementation="cudf")
            self._implementation = "cudf"
        elif hasattr(df, "__narwhals_dataframe__"):  # pragma: no cover
            self._dataframe = df.__narwhals_dataframe__()
            self._implementation = "custom"
        else:
            msg = f"Expected pandas-like dataframe, Polars dataframe, or Polars lazyframe, got: {type(df)}"
            raise TypeError(msg)

    def to_pandas(self) -> Any:
        return self._dataframe.to_pandas()

    def to_numpy(self) -> Any:
        return self._dataframe.to_numpy()

    @property
    def shape(self) -> tuple[int, int]:
        return self._dataframe.shape  # type: ignore[no-any-return]

    def __getitem__(self, col_name: str) -> Series:
        from narwhals.series import Series

        return Series(self._dataframe[col_name], implementation=self._implementation)

    def to_dict(self, *, as_series: bool = True) -> dict[str, Any]:
        return self._dataframe.to_dict(as_series=as_series)  # type: ignore[no-any-return]


class LazyFrame(BaseFrame):
    def __init__(
        self,
        df: Any,
        *,
        implementation: str | None = None,
    ) -> None:
        if implementation is not None:
            self._dataframe: Any = df
            self._implementation = implementation
            return
        if (pl := get_polars()) is not None and isinstance(
            df, (pl.DataFrame, pl.LazyFrame)
        ):
            self._dataframe = df.lazy()
            self._implementation = "polars"
        elif (pd := get_pandas()) is not None and isinstance(df, pd.DataFrame):
            self._dataframe = PandasDataFrame(df, implementation="pandas")
            self._implementation = "pandas"
        elif (mpd := get_modin()) is not None and isinstance(
            df, mpd.DataFrame
        ):  # pragma: no cover
            self._dataframe = PandasDataFrame(df, implementation="modin")
            self._implementation = "modin"
        elif (cudf := get_cudf()) is not None and isinstance(
            df, cudf.DataFrame
        ):  # pragma: no cover
            self._dataframe = PandasDataFrame(df, implementation="cudf")
            self._implementation = "cudf"
        else:
            msg = f"Expected pandas-like dataframe, Polars dataframe, or Polars lazyframe, got: {type(df)}"
            raise TypeError(msg)

    def collect(self) -> DataFrame:
        return DataFrame(
            self._dataframe.collect(),
            implementation=self._implementation,
        )
