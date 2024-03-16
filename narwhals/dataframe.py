from __future__ import annotations

from typing import TYPE_CHECKING
from typing import Any
from typing import Generic
from typing import Iterable
from typing import Literal
from typing import Sequence

from narwhals.dtypes import to_narwhals_dtype
from narwhals.pandas_like.dataframe import PandasDataFrame
from narwhals.translate import get_pandas
from narwhals.translate import get_polars

if TYPE_CHECKING:
    from typing_extensions import Self

    from narwhals.dtypes import DType
    from narwhals.group_by import GroupBy
    from narwhals.series import Series
    from narwhals.typing import IntoExpr
    from narwhals.typing import T
from narwhals.typing import T


def _validate_features(df: Any, features: set[str]) -> None:
    if (pl := get_polars()) is not None and isinstance(df, pl.DataFrame):
        df_features = {"eager"}
    elif (pl := get_polars()) is not None and isinstance(df, pl.LazyFrame):
        df_features = {"lazy"}
    else:
        df_features = df._features
    if diff := {f for f in features if f not in df_features}:
        msg = f"Features {diff} not supported by {type(df)} DataFrame"
        raise TypeError(msg)


class DataFrame(Generic[T]):
    def __init__(
        self,
        df: T,
        *,
        features: Iterable[str] | None = None,
        implementation: str | None = None,
    ) -> None:
        self._features: set[str] = set(features) if features is not None else set()
        if implementation is not None:
            self._dataframe: Any = df
            self._implementation = implementation
            return
        if (pl := get_polars()) is not None and isinstance(
            df, (pl.DataFrame, pl.LazyFrame)
        ):
            self._dataframe = df
            self._implementation = "polars"
        elif (pd := get_pandas()) is not None and isinstance(df, pd.DataFrame):
            self._dataframe = PandasDataFrame(df, implementation="pandas")
            self._implementation = "pandas"
        else:
            msg = f"Expected pandas or Polars dataframe or lazyframe, got: {type(df)}"
            raise TypeError(msg)
        _validate_features(self._dataframe, self._features)

    def _from_dataframe(self, df: Any) -> Self:
        # construct, preserving properties
        return self.__class__(
            df,
            features=self._features,
            implementation=self._implementation,
        )

    def _flatten_and_extract(self, *args: Any, **kwargs: Any) -> Any:
        from narwhals.utils import flatten

        args = [self._extract_native(v) for v in flatten(*args)]  # type: ignore[assignment]
        kwargs = {k: self._extract_native(v) for k, v in kwargs.items()}
        return args, kwargs

    def _extract_native(self, arg: Any) -> Any:
        from narwhals.expression import Expr
        from narwhals.series import Series

        if self._implementation != "polars":
            return arg
        if isinstance(arg, DataFrame):
            return arg._dataframe
        if isinstance(arg, Series):
            return arg._series
        if isinstance(arg, Expr):
            import polars as pl

            return arg._call(pl)
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

    @property
    def shape(self) -> tuple[int, int]:
        if "eager" not in self._features:
            raise RuntimeError(
                "`DataFrame.shape` can only be called when feature 'eager' is enabled"
            )
        return self._dataframe.shape  # type: ignore[no-any-return]

    def __getitem__(self, col_name: str) -> Series[Any]:
        from narwhals.series import Series

        if "eager" not in self._features:
            raise RuntimeError(
                "`DataFrame.__getitem__` can only be called when feature 'eager' is enabled"
            )
        return Series(self._dataframe[col_name], implementation=self._implementation)

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

    def filter(self, *predicates: IntoExpr | Iterable[IntoExpr]) -> Self:
        predicates, _ = self._flatten_and_extract(*predicates)
        return self._from_dataframe(
            self._dataframe.filter(*predicates),
        )

    def group_by(self, *keys: str | Iterable[str]) -> GroupBy[T]:
        from narwhals.group_by import GroupBy

        return GroupBy(self, *keys)

    def sort(
        self,
        by: str | Iterable[str],
        *more_by: str,
        descending: bool | Sequence[bool] = False,
    ) -> Self:
        return self._from_dataframe(
            self._dataframe.sort(by, *more_by, descending=descending)
        )

    def collect(self) -> Self:
        if "lazy" not in self._features:
            raise RuntimeError(
                "`DataFrame.collect` can only be called when feature 'lazy' is enabled"
            )
        features = {f for f in self._features if f != "lazy"}
        features.add("eager")
        return self.__class__(
            self._dataframe.collect(),
            implementation=self._implementation,
            features=features,
        )

    def to_dict(self, *, as_series: bool = True) -> dict[str, Any]:
        return self._dataframe.to_dict(as_series=as_series)  # type: ignore[no-any-return]

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
                other._dataframe,
                how=how,
                left_on=left_on,
                right_on=right_on,
            )
        )

    def to_pandas(self) -> Any:
        if "eager" not in self._features:
            raise RuntimeError(
                "`DataFrame.to_pandas` can only be called when feature 'eager' is enabled"
            )
        return self._dataframe.to_pandas()

    def to_numpy(self) -> Any:
        if "eager" not in self._features:
            raise RuntimeError(
                "`DataFrame.to_numpy` can only be called when feature 'eager' is enabled"
            )
        return self._dataframe.to_numpy()
