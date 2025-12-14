from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

from narwhals._plan.compliant.typing import (
    DataFrameT_co,
    ExprT_co,
    FrameT_contra,
    SeriesT_co,
)

if TYPE_CHECKING:
    from narwhals._plan.expressions import FunctionExpr as FExpr, lists, strings
    from narwhals._plan.expressions.categorical import GetCategories
    from narwhals._plan.expressions.struct import FieldByName
    from narwhals.schema import Schema


class ExprCatNamespace(Protocol[FrameT_contra, ExprT_co]):
    def get_categories(
        self, node: FExpr[GetCategories], frame: FrameT_contra, name: str
    ) -> ExprT_co: ...


class ExprListNamespace(Protocol[FrameT_contra, ExprT_co]):
    def contains(
        self, node: FExpr[lists.Contains], frame: FrameT_contra, name: str
    ) -> ExprT_co: ...
    def get(
        self, node: FExpr[lists.Get], frame: FrameT_contra, name: str
    ) -> ExprT_co: ...
    def len(
        self, node: FExpr[lists.Len], frame: FrameT_contra, name: str
    ) -> ExprT_co: ...
    def unique(
        self, node: FExpr[lists.Unique], frame: FrameT_contra, name: str
    ) -> ExprT_co: ...
    def join(
        self, node: FExpr[lists.Join], frame: FrameT_contra, name: str
    ) -> ExprT_co: ...


class ExprStringNamespace(Protocol[FrameT_contra, ExprT_co]):
    def contains(
        self, node: FExpr[strings.Contains], frame: FrameT_contra, name: str
    ) -> ExprT_co: ...
    def ends_with(
        self, node: FExpr[strings.EndsWith], frame: FrameT_contra, name: str
    ) -> ExprT_co: ...
    def len_chars(
        self, node: FExpr[strings.LenChars], frame: FrameT_contra, name: str
    ) -> ExprT_co: ...
    def replace(
        self, node: FExpr[strings.Replace], frame: FrameT_contra, name: str
    ) -> ExprT_co: ...
    def replace_all(
        self, node: FExpr[strings.ReplaceAll], frame: FrameT_contra, name: str
    ) -> ExprT_co: ...
    def slice(
        self, node: FExpr[strings.Slice], frame: FrameT_contra, name: str
    ) -> ExprT_co: ...
    def split(
        self, node: FExpr[strings.Split], frame: FrameT_contra, name: str
    ) -> ExprT_co: ...
    def starts_with(
        self, node: FExpr[strings.StartsWith], frame: FrameT_contra, name: str
    ) -> ExprT_co: ...
    def strip_chars(
        self, node: FExpr[strings.StripChars], frame: FrameT_contra, name: str
    ) -> ExprT_co: ...
    def to_uppercase(
        self, node: FExpr[strings.ToUppercase], frame: FrameT_contra, name: str
    ) -> ExprT_co: ...
    def to_lowercase(
        self, node: FExpr[strings.ToLowercase], frame: FrameT_contra, name: str
    ) -> ExprT_co: ...
    def to_titlecase(
        self, node: FExpr[strings.ToTitlecase], frame: FrameT_contra, name: str
    ) -> ExprT_co: ...
    def to_date(
        self, node: FExpr[strings.ToDate], frame: FrameT_contra, name: str
    ) -> ExprT_co: ...
    def to_datetime(
        self, node: FExpr[strings.ToDatetime], frame: FrameT_contra, name: str
    ) -> ExprT_co: ...
    def zfill(
        self, node: FExpr[strings.ZFill], frame: FrameT_contra, name: str
    ) -> ExprT_co: ...


class ExprStructNamespace(Protocol[FrameT_contra, ExprT_co]):
    def field(
        self, node: FExpr[FieldByName], frame: FrameT_contra, name: str
    ) -> ExprT_co: ...


class SeriesStructNamespace(Protocol[DataFrameT_co, SeriesT_co]):
    def field(self, name: str) -> SeriesT_co: ...
    def unnest(self) -> DataFrameT_co: ...
    @property
    def schema(self) -> Schema: ...
