from __future__ import annotations

import re
from functools import partial
from typing import TYPE_CHECKING

from narwhals._spark_like.utils import strptime_to_pyspark_format
from narwhals._sql.expr_str import SQLExprStringNamespace
from narwhals._utils import _is_naive_format, not_implemented, requires

if TYPE_CHECKING:
    from sqlframe.base.column import Column

    from narwhals._spark_like.expr import SparkLikeExpr


class SparkLikeExprStringNamespace(SQLExprStringNamespace["SparkLikeExpr"]):
    def _strip_chars(self, characters: str, *, start: bool) -> SparkLikeExpr:
        escaped = re.escape(characters)
        pattern = rf"^[{escaped}]+" if start else rf"[{escaped}]+$"
        F = self.compliant._F
        return self.compliant._with_elementwise(
            lambda expr: F.regexp_replace(expr, pattern, "")
        )

    def strip_chars_start(self, characters: str) -> SparkLikeExpr:
        return self._strip_chars(characters, start=True)

    def strip_chars_end(self, characters: str) -> SparkLikeExpr:
        return self._strip_chars(characters, start=False)

    def to_datetime(self, format: str | None) -> SparkLikeExpr:
        F = self.compliant._F
        if not format:
            function = F.to_timestamp
        elif _is_naive_format(format):
            function = partial(
                F.to_timestamp_ntz, format=F.lit(strptime_to_pyspark_format(format))
            )
        else:
            format = strptime_to_pyspark_format(format)
            function = partial(F.to_timestamp, format=format)
        return self.compliant._with_elementwise(
            lambda expr: function(F.replace(expr, F.lit("T"), F.lit(" ")))
        )

    def to_date(self, format: str | None) -> SparkLikeExpr:
        F = self.compliant._F
        return self.compliant._with_elementwise(
            lambda expr: F.to_date(expr, format=strptime_to_pyspark_format(format))
        )

    def to_time(self, format: str | None) -> SparkLikeExpr:
        msg = "spark-like backends do not support the Time type"
        raise ValueError(msg)

    def to_titlecase(self) -> SparkLikeExpr:
        impl = self.compliant._implementation
        sqlframe_required_version = (3, 43, 1)
        if (
            impl.is_sqlframe()
            and (version := impl._backend_version()) < sqlframe_required_version
        ):  # pragma: no cover
            required_str = requires._unparse_version(sqlframe_required_version)
            found_str = requires._unparse_version(version)
            msg = (
                f"`str.to_titlecase` is only available in 'sqlframe>={required_str}', "
                f"found version {found_str!r}."
            )
            raise NotImplementedError(msg)

        def _to_titlecase(expr: Column) -> Column:
            F = self.compliant._F
            lower_expr = F.lower(expr)
            extract_expr = F.regexp_extract_all(
                lower_expr, regexp=F.lit(r"[a-z]*[^a-z]*"), idx=0
            )
            capitalized_expr = F.transform(extract_expr, f=F.initcap)
            return F.array_join(capitalized_expr, delimiter="")

        return self.compliant._with_elementwise(_to_titlecase)

    replace = not_implemented()
