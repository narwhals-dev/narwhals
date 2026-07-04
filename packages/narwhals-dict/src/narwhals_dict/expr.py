from __future__ import annotations

from typing import TYPE_CHECKING

from narwhals._compliant import EagerExpr
from narwhals._utils import Implementation, not_implemented
from narwhals_dict.series import DictSeries

if TYPE_CHECKING:
    from typing_extensions import Self

    from narwhals._compliant.typing import AliasNames, EvalNames, EvalSeries
    from narwhals._utils import Version, _LimitedContext
    from narwhals_dict.dataframe import DictDataFrame
    from narwhals_dict.namespace import DictNamespace


class DictExpr(EagerExpr["DictDataFrame", DictSeries]):
    _implementation: Implementation = Implementation.UNKNOWN

    def __init__(
        self,
        call: EvalSeries[DictDataFrame, DictSeries],
        *,
        evaluate_output_names: EvalNames[DictDataFrame],
        alias_output_names: AliasNames | None,
        version: Version,
        implementation: Implementation = Implementation.UNKNOWN,
    ) -> None:
        self._call = call
        self._evaluate_output_names = evaluate_output_names
        self._alias_output_names = alias_output_names
        self._version = version

    @classmethod
    def from_column_names(
        cls,
        evaluate_column_names: EvalNames[DictDataFrame],
        /,
        *,
        context: _LimitedContext,
    ) -> Self:
        def func(df: DictDataFrame) -> list[DictSeries]:
            try:
                return [
                    DictSeries(df.native[name], name=name, version=df._version)
                    for name in evaluate_column_names(df)
                ]
            except KeyError as e:
                if error := df._check_columns_exist(evaluate_column_names(df)):
                    raise error from e
                raise

        return cls(
            func,
            evaluate_output_names=evaluate_column_names,
            alias_output_names=None,
            version=context._version,
        )

    @classmethod
    def from_column_indices(cls, *column_indices: int, context: _LimitedContext) -> Self:
        def func(df: DictDataFrame) -> list[DictSeries]:
            columns = df.columns
            return [
                DictSeries(df.native[columns[i]], name=columns[i], version=df._version)
                for i in column_indices
            ]

        return cls(
            func,
            evaluate_output_names=cls._eval_names_indices(column_indices),
            alias_output_names=None,
            version=context._version,
        )

    def __narwhals_namespace__(self) -> DictNamespace:
        from narwhals_dict.namespace import DictNamespace

        return DictNamespace(version=self._version)

    ewm_mean = not_implemented()
    over = not_implemented()
