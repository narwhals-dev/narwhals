import polars as pl
from typing import Iterable
from polars_api_compat.spec import (
    DataFrame as DataFrameT,
    LazyFrame as LazyFrameT,
    Namespace as NamespaceT,
    Expr as ExprT,
    IntoExpr,
    GroupBy as GroupByT,
)


def translate(df: pl.LazyFrame | pl.DataFrame) -> tuple[LazyFrameT, NamespaceT]:
    if isinstance(df, pl.DataFrame):
        return DataFrame(df), Namespace(api_version="2023.11-beta")
    if isinstance(df, pl.LazyFrame):
        return LazyFrame(df), Namespace(api_version="2023.11-beta")
    raise TypeError(
        f"Could not translate DataFrame {type(df)}, please open a feature request."
    )


class DataFrame(DataFrameT):
    def __init__(self, df: pl.DataFrame, *, api_version: str) -> None:
        self.df = df
        self.api_version = api_version
        self.columns = df.columns
        self.dataframe = df

    def with_columns(
        self, *exprs: IntoExpr | Iterable[IntoExpr], **named_exprs: IntoExpr
    ) -> DataFrameT:
        return DataFrame(self.df.with_columns(*exprs, **named_exprs))

    def filter(self, *predicates: IntoExpr | Iterable[IntoExpr]) -> DataFrameT:
        return DataFrame(self.df.filter(*predicates))

    def select(
        self, *exprs: IntoExpr | Iterable[IntoExpr], **named_exprs: IntoExpr
    ) -> DataFrameT:
        return DataFrame(self.df.select(*exprs, **named_exprs))

    @property
    def dataframe(self):
        return self.df

    def __dataframe_namespace__(self) -> NamespaceT:
        return Namespace(api_version=self.api_version)


class LazyFrame(LazyFrameT):
    def __init__(self, df: pl.LazyFrame, *, api_version: str) -> None:
        self.df = df
        self.api_version = api_version

    def with_columns(
        self, *exprs: IntoExpr | Iterable[IntoExpr], **named_exprs: IntoExpr
    ) -> DataFrameT:
        return DataFrame(self.df.with_columns(*exprs, **named_exprs))

    def filter(self, *predicates: IntoExpr | Iterable[IntoExpr]) -> DataFrameT:
        return DataFrame(self.df.filter(*predicates))

    def select(
        self, *exprs: IntoExpr | Iterable[IntoExpr], **named_exprs: IntoExpr
    ) -> DataFrameT:
        return DataFrame(self.df.select(*exprs, **named_exprs))

    @property
    def dataframe(self):
        return self.df

    def __lazyframe_namespace__(self) -> NamespaceT:
        return Namespace(api_version=self.api_version)

    def group_by(self, *keys: str | Iterable[str]) -> GroupByT:
        return GroupBy(self.df.group_by(*keys), api_version=self.api_version)


class GroupBy(GroupByT):
    def __init__(self, df: pl.DataFrame, *keys: str, api_version: str) -> None:
        self.df = df
        self.api_version = api_version
        self._keys = keys

    def agg(
        self,
        *aggs: IntoExpr | Iterable[IntoExpr],
        **named_aggs: IntoExpr,
    ) -> DataFrameT:
        return DataFrame(
            self.df.group_by(self._keys).agg(*aggs, **named_aggs),
            api_version=self.api_version,
        )


class Namespace(NamespaceT):
    def __init__(self, *, api_version: str) -> None:
        self.api_version = api_version

    def col(self, *column_names: str | Iterable[str]) -> ExprT:
        return Expr(pl.col(*column_names))


class Expr(ExprT):
    def __init__(self, expr: pl.Expr) -> None:
        self.expr = expr
