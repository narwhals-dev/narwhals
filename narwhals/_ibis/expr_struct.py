from __future__ import annotations

from typing import TYPE_CHECKING, cast

from narwhals._compliant import LazyExprNamespace
from narwhals._compliant.any_namespace import StructNamespace

if TYPE_CHECKING:
    import ibis.expr.types as ir
    from ibis.expr.datatypes import Struct as StructDtype

    from narwhals._ibis.dataframe import IbisLazyFrame
    from narwhals._ibis.expr import IbisExpr


class IbisExprStructNamespace(LazyExprNamespace["IbisExpr"], StructNamespace["IbisExpr"]):
    def field(self, name: str) -> IbisExpr:
        def func(expr: ir.StructColumn) -> ir.Column:
            return expr[name]

        return self.compliant._with_callable(func).alias(name)

    def unnest(self) -> IbisExpr:
        compliant = self.compliant

        def func(df: IbisLazyFrame) -> list[ir.Column]:
            schema = df.schema
            return [
                cast("ir.StructColumn", native_expr)[field.name].name(field.name)
                for native_expr, name in zip(
                    compliant(df), compliant._evaluate_output_names(df)
                )
                for field in cast("StructDtype", schema[name]).fields
            ]

        def evaluate_output_names(df: IbisLazyFrame) -> list[str]:
            schema = df.schema
            return [
                field.name
                for name in compliant._evaluate_output_names(df)
                for field in cast("StructDtype", schema[name]).fields
            ]

        return compliant.__class__(
            func,
            evaluate_output_names=evaluate_output_names,
            alias_output_names=None,
            version=compliant._version,
            implementation=compliant._implementation,
        )
