"""Researching logical plan implementations.

## Why?
`polars` has multiple *logical plan* representations, including [`DslPlan`] (main focus), [`IR`], [`IRPlan`].

The motivation for having a `LogicalPlan` in `narwhals` is to better support rewriting expressions and queries.

- **Writing a query engine is a non-goal**.
- **There will not be `{Execution,Physical}Plan`(s)**.

It is important to see more examples (particularly high-level ones) of usage - to get a feel for things we do and don't need.
Blindly copying from `polars` for this is likely to pull in things that aren't helpful for our use case.

## Projects
- [`datafusion`]
  - https://docs.rs/datafusion-expr/latest/datafusion_expr/logical_plan/enum.LogicalPlan.html#variants
- [`sqlglot`]
  - https://github.com/tobymao/sqlglot/blob/b93291bee7ada32b4d686db919d8d3d683d95425/sqlglot/planner.py
  - https://github.com/tobymao/sqlglot/blob/b93291bee7ada32b4d686db919d8d3d683d95425/sqlglot/expressions.py
- [`ibis`]
- [`cudf_polars`]


[`DslPlan`]: https://github.com/pola-rs/polars/blob/46bb9cc95a7dba3603a1fcd9d2a955040102f5a2/crates/polars-plan/src/dsl/plan.rs#L28-L177
[`IR`]: https://github.com/pola-rs/polars/blob/46bb9cc95a7dba3603a1fcd9d2a955040102f5a2/crates/polars-plan/src/plans/ir/mod.rs#L38-L166
[`IRPlan`]: https://github.com/pola-rs/polars/blob/46bb9cc95a7dba3603a1fcd9d2a955040102f5a2/crates/polars-plan/src/plans/ir/mod.rs#L168-L204
[`datafusion`]: https://datafusion.apache.org/library-user-guide/building-logical-plans.html
[`sqlglot`]: https://github.com/tobymao/sqlglot/blob/b93291bee7ada32b4d686db919d8d3d683d95425/posts/ast_primer.md
[`ibis`]: https://github.com/ibis-project/ibis/blob/9126733b38e1c92f6e787f92dc9954e88ab6400d/ibis/expr/operations/relations.py
[`cudf_polars`]: https://github.com/rapidsai/cudf/blob/e3b3ac371aa67d91a812dec4645c972279de866b/python/cudf_polars/cudf_polars/dsl/ir.py
"""

from __future__ import annotations

from narwhals._plan.logical_plan.builder import LpBuilder as LpBuilder

__all__ = ["LpBuilder"]
