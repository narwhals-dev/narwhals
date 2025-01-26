from __future__ import annotations

import re
from functools import partial
from typing import TYPE_CHECKING
from typing import Any
from typing import Callable
from typing import Sequence

from pyspark.sql import functions as F  # noqa: N812

from narwhals._expression_parsing import is_simple_aggregation
from narwhals._spark_like.utils import _std
from narwhals._spark_like.utils import _var
from narwhals.utils import parse_version

if TYPE_CHECKING:
    from pyspark.sql import Column
    from pyspark.sql import GroupedData
    from typing_extensions import Self

    from narwhals._spark_like.dataframe import SparkLikeLazyFrame
    from narwhals._spark_like.typing import SparkLikeExpr
    from narwhals.typing import CompliantExpr


class SparkLikeLazyGroupBy:
    def __init__(
        self: Self,
        df: SparkLikeLazyFrame,
        keys: list[str],
        drop_null_keys: bool,  # noqa: FBT001
    ) -> None:
        self._df = df
        self._keys = keys
        if drop_null_keys:
            self._grouped = self._df._native_frame.dropna(subset=self._keys).groupBy(
                *self._keys
            )
        else:
            self._grouped = self._df._native_frame.groupBy(*self._keys)

    def agg(
        self: Self,
        *exprs: SparkLikeExpr,
    ) -> SparkLikeLazyFrame:
        return agg_pyspark(
            self._df,
            self._grouped,
            exprs,
            self._keys,
            self._from_native_frame,
        )

    def _from_native_frame(self: Self, df: SparkLikeLazyFrame) -> SparkLikeLazyFrame:
        from narwhals._spark_like.dataframe import SparkLikeLazyFrame

        return SparkLikeLazyFrame(
            df, backend_version=self._df._backend_version, version=self._df._version
        )


def get_spark_function(function_name: str, **kwargs: Any) -> Column:
    if function_name in {"std", "var"}:
        import numpy as np  # ignore-banned-import

        return partial(
            _std if function_name == "std" else _var,
            ddof=kwargs["ddof"],
            np_version=parse_version(np.__version__),
        )

    elif function_name == "len":
        # Use count(*) to count all rows including nulls
        def _count(*_args: Any, **_kwargs: Any) -> Column:
            return F.count("*")

        return _count

    elif function_name == "n_unique":
        from pyspark.sql.types import IntegerType

        def _n_unique(_input: Column) -> Column:
            return F.count_distinct(_input) + F.max(F.isnull(_input).cast(IntegerType()))

        return _n_unique

    else:
        return getattr(F, function_name)


def agg_pyspark(
    df: SparkLikeLazyFrame,
    grouped: GroupedData,
    exprs: Sequence[CompliantExpr[Column]],
    keys: list[str],
    from_dataframe: Callable[[Any], SparkLikeLazyFrame],
) -> SparkLikeLazyFrame:
    if not exprs:
        # No aggregation provided
        return from_dataframe(df._native_frame.select(*keys).dropDuplicates(subset=keys))

    for expr in exprs:
        if not is_simple_aggregation(expr):  # pragma: no cover
            msg = (
                "Non-trivial complex aggregation found.\n\n"
                "Hint: you were probably trying to apply a non-elementary aggregation with a "
                "dask dataframe.\n"
                "Please rewrite your query such that group-by aggregations "
                "are elementary. For example, instead of:\n\n"
                "    df.group_by('a').agg(nw.col('b').round(2).mean())\n\n"
                "use:\n\n"
                "    df.with_columns(nw.col('b').round(2)).group_by('a').agg(nw.col('b').mean())\n\n"
            )
            raise ValueError(msg)

    simple_aggregations: dict[str, Column] = {}
    for expr in exprs:
        output_names = expr._evaluate_output_names(df)
        aliases = (
            output_names
            if expr._alias_output_names is None
            else expr._alias_output_names(output_names)
        )
        if len(output_names) > 1:
            # For multi-output aggregations, e.g. `df.group_by('a').agg(nw.all().mean())`, we skip
            # the keys, else they would appear duplicated in the output.
            output_names, aliases = zip(
                *[(x, alias) for x, alias in zip(output_names, aliases) if x not in keys]
            )
        if expr._depth == 0:  # pragma: no cover
            # e.g. agg(nw.len()) # noqa: ERA001
            agg_func = get_spark_function(expr._function_name, **expr._kwargs)
            simple_aggregations.update({alias: agg_func(keys[0]) for alias in aliases})
            continue

        # e.g. agg(nw.mean('a')) # noqa: ERA001
        function_name = re.sub(r"(\w+->)", "", expr._function_name)
        agg_func = get_spark_function(function_name, **expr._kwargs)

        simple_aggregations.update(
            {
                alias: agg_func(output_name)
                for alias, output_name in zip(aliases, output_names)
            }
        )

    agg_columns = [col_.alias(name) for name, col_ in simple_aggregations.items()]
    result_simple = grouped.agg(*agg_columns)
    return from_dataframe(result_simple)
