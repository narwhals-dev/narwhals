from __future__ import annotations

import collections
from typing import TYPE_CHECKING
from typing import Any
from typing import ClassVar
from typing import Iterator
from typing import Mapping
from typing import Sequence

import pyarrow as pa
import pyarrow.compute as pc

from narwhals._arrow.utils import cast_to_comparable_string_types
from narwhals._arrow.utils import extract_py_scalar
from narwhals._compliant import EagerGroupBy
from narwhals._expression_parsing import evaluate_output_names_and_aliases
from narwhals.utils import generate_temporary_column_name

if TYPE_CHECKING:
    from typing_extensions import Self

    from narwhals._arrow.dataframe import ArrowDataFrame
    from narwhals._arrow.expr import ArrowExpr
    from narwhals._arrow.typing import Incomplete
    from narwhals._compliant.group_by import NarwhalsAggregation


class ArrowGroupBy(EagerGroupBy["ArrowDataFrame", "ArrowExpr"]):
    _REMAP_AGGS: ClassVar[Mapping[NarwhalsAggregation, Any]] = {
        "sum": "sum",
        "mean": "mean",
        "median": "approximate_median",
        "max": "max",
        "min": "min",
        "std": "stddev",
        "var": "variance",
        "len": "count",
        "n_unique": "count_distinct",
        "count": "count",
    }

    def __init__(
        self,
        compliant_frame: ArrowDataFrame,
        keys: Sequence[str],
        /,
        *,
        drop_null_keys: bool,
    ) -> None:
        if drop_null_keys:
            self._compliant_frame = compliant_frame.drop_nulls(keys)
        else:
            self._compliant_frame = compliant_frame
        self._keys: list[str] = list(keys)
        self._grouped = pa.TableGroupBy(self.compliant.native, self._keys)

    def agg(self: Self, *exprs: ArrowExpr) -> ArrowDataFrame:
        self._ensure_all_simple(exprs)
        aggs: list[tuple[str, str, Any]] = []
        expected_pyarrow_column_names: list[str] = self._keys.copy()
        new_column_names: list[str] = self._keys.copy()

        for expr in exprs:
            output_names, aliases = evaluate_output_names_and_aliases(
                expr, self.compliant, self._keys
            )

            if expr._depth == 0:
                # e.g. `agg(nw.len())`
                if expr._function_name != "len":  # pragma: no cover
                    msg = "Safety assertion failed, please report a bug to https://github.com/narwhals-dev/narwhals/issues"
                    raise AssertionError(msg)

                new_column_names.append(aliases[0])
                expected_pyarrow_column_names.append(f"{self._keys[0]}_count")
                aggs.append((self._keys[0], "count", pc.CountOptions(mode="all")))
                continue

            function_name = self._leaf_name(expr)
            if function_name in {"std", "var"}:
                option: Any = pc.VarianceOptions(ddof=expr._call_kwargs["ddof"])
            elif function_name in {"len", "n_unique"}:
                option = pc.CountOptions(mode="all")
            elif function_name == "count":
                option = pc.CountOptions(mode="only_valid")
            else:
                option = None

            function_name = self._remap_expr_name(function_name)
            new_column_names.extend(aliases)
            expected_pyarrow_column_names.extend(
                [f"{output_name}_{function_name}" for output_name in output_names]
            )
            aggs.extend(
                [(output_name, function_name, option) for output_name in output_names]
            )

        result_simple = self._grouped.aggregate(aggs)

        # Rename columns, being very careful
        expected_old_names_indices: dict[str, list[int]] = collections.defaultdict(list)
        for idx, item in enumerate(expected_pyarrow_column_names):
            expected_old_names_indices[item].append(idx)
        if not (
            set(result_simple.column_names) == set(expected_pyarrow_column_names)
            and len(result_simple.column_names) == len(expected_pyarrow_column_names)
        ):  # pragma: no cover
            msg = (
                f"Safety assertion failed, expected {expected_pyarrow_column_names} "
                f"got {result_simple.column_names}, "
                "please report a bug at https://github.com/narwhals-dev/narwhals/issues"
            )
            raise AssertionError(msg)
        index_map: list[int] = [
            expected_old_names_indices[item].pop(0) for item in result_simple.column_names
        ]
        new_column_names = [new_column_names[i] for i in index_map]
        result_simple = result_simple.rename_columns(new_column_names)
        if self.compliant._backend_version < (12, 0, 0):
            columns = result_simple.column_names
            result_simple = result_simple.select(
                [*self._keys, *[col for col in columns if col not in self._keys]]
            )
        return self.compliant._with_native(result_simple)

    def __iter__(self: Self) -> Iterator[tuple[Any, ArrowDataFrame]]:
        col_token = generate_temporary_column_name(
            n_bytes=8, columns=self.compliant.columns
        )
        null_token: str = "__null_token_value__"  # noqa: S105

        table = self.compliant.native
        it, separator_scalar = cast_to_comparable_string_types(
            *(table[key] for key in self._keys), separator=""
        )
        # NOTE: stubs indicate `separator` must also be a `ChunkedArray`
        # Reality: `str` is fine
        concat_str: Incomplete = pc.binary_join_element_wise
        key_values = concat_str(
            *it,
            separator_scalar,
            null_handling="replace",
            null_replacement=null_token,
        )
        table = table.add_column(i=0, field_=col_token, column=key_values)
        for v in pc.unique(key_values):
            t = self.compliant._with_native(
                table.filter(pc.equal(table[col_token], v)).drop([col_token])
            )
            row = t.simple_select(*self._keys).row(0)
            yield tuple(extract_py_scalar(el) for el in row), t
