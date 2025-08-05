from __future__ import annotations

from typing import Any, Generic

from narwhals._compliant import LazyExprNamespace
from narwhals._compliant.any_namespace import StringNamespace
from narwhals._sql.typing import SQLExprT


class SQLExprStringNamespace(
    LazyExprNamespace["SQLExprT"], StringNamespace["SQLExprT"], Generic[SQLExprT]
):
    def _lit(self, value: Any) -> SQLExprT:
        return self.compliant._lit(value)

    def _function(self, name: str, *args: Any) -> SQLExprT:
        return self.compliant._function(name, *args)

    def _when(self, *args: Any, **kwargs: Any) -> SQLExprT:
        return self.compliant._when(*args, **kwargs)

    def to_lowercase(self) -> SQLExprT:
        return self.compliant._with_elementwise(
            lambda expr: self._function("lower", expr)
        )

    def to_uppercase(self) -> SQLExprT:
        return self.compliant._with_elementwise(
            lambda expr: self._function("upper", expr)
        )

    def starts_with(self, prefix: str) -> SQLExprT:
        return self.compliant._with_elementwise(
            lambda expr: self._function("starts_with", expr, self._lit(prefix))
        )

    def ends_with(self, suffix: str) -> SQLExprT:
        return self.compliant._with_elementwise(
            lambda expr: self._function("ends_with", expr, self._lit(suffix))
        )

    def zfill(self, width: int) -> SQLExprT:
        # There is no built-in zfill function, so we need to implement it manually
        # using string manipulation functions.

        def func(expr: Any) -> Any:
            less_than_width = self._function("length", expr) < self._lit(width)
            zero, hyphen, plus = self._lit("0"), self._lit("-"), self._lit("+")

            starts_with_minus = self._function("starts_with", expr, hyphen)
            starts_with_plus = self._function("starts_with", expr, plus)
            substring = self._function("substr", expr, self._lit(2))
            padded_substring = self._function(
                "lpad", substring, self._lit(width - 1), zero
            )
            return self._when(
                starts_with_minus & less_than_width,
                self._function("concat", hyphen, padded_substring),
                self._when(
                    starts_with_plus & less_than_width,
                    self._function("concat", plus, padded_substring),
                    self._when(
                        less_than_width,
                        self._function("lpad", expr, self._lit(width), zero),
                        expr,
                    ),
                ),
            )

        # can't use `_with_elementwise` due to `when` operator.
        # TODO(unassigned): implement `window_func` like we do in `Expr.cast`
        return self.compliant._with_callable(func)
