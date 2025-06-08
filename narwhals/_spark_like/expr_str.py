from __future__ import annotations

from functools import partial
from typing import TYPE_CHECKING

from narwhals._spark_like.utils import strptime_to_pyspark_format
from narwhals._utils import _is_naive_format

if TYPE_CHECKING:
    from sqlframe.base.column import Column

    from narwhals._spark_like.expr import SparkLikeExpr


class SparkLikeExprStringNamespace:
    def __init__(self, expr: SparkLikeExpr) -> None:
        self._compliant_expr = expr

    def len_chars(self) -> SparkLikeExpr:
        return self._compliant_expr._with_callable(self._compliant_expr._F.char_length)

    def replace_all(self, pattern: str, value: str, *, literal: bool) -> SparkLikeExpr:
        def func(expr: Column) -> Column:
            replace_all_func = (
                self._compliant_expr._F.replace
                if literal
                else self._compliant_expr._F.regexp_replace
            )
            return replace_all_func(
                expr,
                self._compliant_expr._F.lit(pattern),  # pyright: ignore[reportArgumentType]
                self._compliant_expr._F.lit(value),  # pyright: ignore[reportArgumentType]
            )

        return self._compliant_expr._with_callable(func)

    def strip_chars(self, characters: str | None) -> SparkLikeExpr:
        import string

        def func(expr: Column) -> Column:
            to_remove = characters if characters is not None else string.whitespace
            return self._compliant_expr._F.btrim(
                expr, self._compliant_expr._F.lit(to_remove)
            )

        return self._compliant_expr._with_callable(func)

    def starts_with(self, prefix: str) -> SparkLikeExpr:
        return self._compliant_expr._with_callable(
            lambda expr: self._compliant_expr._F.startswith(
                expr, self._compliant_expr._F.lit(prefix)
            )
        )

    def ends_with(self, suffix: str) -> SparkLikeExpr:
        return self._compliant_expr._with_callable(
            lambda expr: self._compliant_expr._F.endswith(
                expr, self._compliant_expr._F.lit(suffix)
            )
        )

    def contains(self, pattern: str, *, literal: bool) -> SparkLikeExpr:
        def func(expr: Column) -> Column:
            contains_func = (
                self._compliant_expr._F.contains
                if literal
                else self._compliant_expr._F.regexp
            )
            return contains_func(expr, self._compliant_expr._F.lit(pattern))

        return self._compliant_expr._with_callable(func)

    def slice(self, offset: int, length: int | None) -> SparkLikeExpr:
        # From the docs: https://spark.apache.org/docs/latest/api/python/reference/pyspark.sql/api/pyspark.sql.functions.substring.html
        # The position is not zero based, but 1 based index.
        def func(expr: Column) -> Column:
            col_length = self._compliant_expr._F.char_length(expr)

            _offset = (
                col_length + self._compliant_expr._F.lit(offset + 1)
                if offset < 0
                else self._compliant_expr._F.lit(offset + 1)
            )
            _length = (
                self._compliant_expr._F.lit(length) if length is not None else col_length
            )
            return expr.substr(_offset, _length)

        return self._compliant_expr._with_callable(func)

    def split(self, by: str) -> SparkLikeExpr:
        return self._compliant_expr._with_callable(
            lambda expr: self._compliant_expr._F.split(expr, by)
        )

    def to_uppercase(self) -> SparkLikeExpr:
        return self._compliant_expr._with_callable(self._compliant_expr._F.upper)

    def to_lowercase(self) -> SparkLikeExpr:
        return self._compliant_expr._with_callable(self._compliant_expr._F.lower)

    def to_datetime(self, format: str | None) -> SparkLikeExpr:
        F = self._compliant_expr._F  # noqa: N806
        if not format:
            function = F.to_timestamp
        elif _is_naive_format(format):
            function = partial(
                F.to_timestamp_ntz, format=F.lit(strptime_to_pyspark_format(format))
            )
        else:
            format = strptime_to_pyspark_format(format)
            function = partial(F.to_timestamp, format=format)
        return self._compliant_expr._with_callable(
            lambda expr: function(F.replace(expr, F.lit("T"), F.lit(" ")))
        )
