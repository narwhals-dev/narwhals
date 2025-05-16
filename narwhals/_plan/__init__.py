"""Brainstorming an `Expr` internal represention.

Notes:
- Each `Expr` method should be representable by a single node
  - But the node does not need to be unique to the method
- A chain of `Expr` methods should form a plan of operations
- We must be able to enforce rules on what plans are permitted:
  - Must be flexible to both eager/lazy and individual backends
  - Must be flexible to a given context (select, with_columns, filter, group_by)
- Nodes & plans are:
  - Immutable, but
    - Can be extended/re-written at both the Narwhals & Compliant levels
  - Introspectable, but
    - Store as little-as-needed for the common case
    - Provide properties/methods for computing the less frequent metadata

References:
- https://github.com/pola-rs/polars/blob/dafd0a2d0e32b52bcfa4273bffdd6071a0d5977a/crates/polars-python/src/lazyframe/visitor/expr_nodes.rs
- https://github.com/pola-rs/polars/blob/dafd0a2d0e32b52bcfa4273bffdd6071a0d5977a/crates/polars-plan/src/dsl/expr.rs
- https://github.com/pola-rs/polars/blob/dafd0a2d0e32b52bcfa4273bffdd6071a0d5977a/crates/polars-plan/src/dsl/function_expr/mod.rs
- https://github.com/pola-rs/polars/blob/dafd0a2d0e32b52bcfa4273bffdd6071a0d5977a/crates/polars-plan/src/dsl/options/mod.rs#L137-L172
- https://github.com/pola-rs/polars/blob/3fd7ecc5f9de95f62b70ea718e7e5dbf951b6d1c/crates/polars-plan/src/plans/options.rs#L35-L106
- https://github.com/pola-rs/polars/blob/3fd7ecc5f9de95f62b70ea718e7e5dbf951b6d1c/crates/polars-plan/src/plans/options.rs#L131-L236
- https://github.com/pola-rs/polars/blob/3fd7ecc5f9de95f62b70ea718e7e5dbf951b6d1c/crates/polars-plan/src/plans/options.rs#L240-L267
- https://github.com/pola-rs/polars/blob/6df23a09a81c640c21788607611e09d9f43b1abc/crates/polars-plan/src/plans/aexpr/mod.rs

Related:
- https://github.com/narwhals-dev/narwhals/pull/2483#issuecomment-2866902903
- https://github.com/narwhals-dev/narwhals/pull/2483#issuecomment-2867331343
- https://github.com/narwhals-dev/narwhals/pull/2483#issuecomment-2867446959
- https://github.com/narwhals-dev/narwhals/pull/2483#issuecomment-2869070157
- (https://github.com/narwhals-dev/narwhals/pull/2538/commits/a7eeb0d23e67cb70e7cfa73cec2c7b69a15c8bef#r2083562677)
- https://github.com/narwhals-dev/narwhals/issues/2225
- https://github.com/narwhals-dev/narwhals/issues/1848
- https://github.com/narwhals-dev/narwhals/issues/2534#issuecomment-2875676729
- https://github.com/narwhals-dev/narwhals/issues/2291
- https://github.com/narwhals-dev/narwhals/issues/2522
"""

from __future__ import annotations
