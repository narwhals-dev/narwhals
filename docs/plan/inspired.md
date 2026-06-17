# Inspiration and comparisons

<!-- TODO @dangotbanned: All super rough, but needed to start somewhere -->

Many projects have influenced this one, and here's where you can learn what got us here.


## [Narwhals](../index.md)
This project would have never began without it.

- Defined the base API surface that needed to be reproduced
- Issue tracker, discussions, reviews helped inform where things are difficult/not yet feasible (see [Related issues])
- Code
    - All native re-impls are either verbatim from main or heavily inspired by it
    - Test suite is heavily based on `tests/`
        - In some cases it is a superset and this has uncovered bugs <!-- TODO @dangotbanned: Link bugs -->

[Related issues]: ./related-issues.md#related-issues

## [Polars](https://docs.pola.rs/)
- All of `*IR`, `*Plan` is adapted from the rust dsl
- Borrowed lots of the test suite for new features


<!-- TODO @dangotbanned: Visualize the layers/concepts

- Expr (compare with polars and main)
  - Parse
  - Expansion
  - Resolve (name/dtype)
  - Dispatch (hand over control to compliant)
-->


## [Ibis](https://ibis-project.org/)


### Learned
<!-- TODO @dangotbanned: Need to spend more time on this, it really was helpful -->
- Meta-programming
    - repo is a huge resource for understanding what's possible in python

### Differences

#### Ibis inspects type annotations at runtime
This stuff is challenging to do correctly, especially without depending on [`typing_extensions`][] [^1].

There are also costs at import-time[^2], before we even reach the runtime validation (https://github.com/ibis-project/ibis/pull/11885).

!!! info
    Like the rest of Narwhals, *static typing* is used heavily.  
    However, the focus is on documentation and correctness (yes, in that order)

[^1]: Or (https://github.com/annotated-types/annotated-types) or (https://github.com/pydantic/typing-inspection) or ...
[^2]: Which can be mitigated in some cases with [`annotationlib`][], but not when they're [evaluated] for [each new subclass]

[evaluated]: https://github.com/ibis-project/ibis/blob/50775279a4ba5f312202ff522bd5b0b4b9683d40/ibis/common/typing.py#L163-L206
[each new subclass]: https://github.com/ibis-project/ibis/blob/50775279a4ba5f312202ff522bd5b0b4b9683d40/ibis/common/grounds.py#L51-L54

#### Misc
- Don't have that exprs vs operations concept
- Compiling to SQL is an implementation detail for some backends (not a general goal)

## [SQLGlot](https://sqlglot.com/sqlglot.html)
<!-- TODO @dangotbanned: `sqlglot.expressions.ExplodingGenerateSeries`

See write-up in (/GitHub/narwhals/plan-narwhals-exploding_generate_series.md)
-->
- Early version of expression dispatch adapted an idea from [`sqlglot.Generator`](https://github.com/tobymao/sqlglot/blob/75ffde8a90344e5047f962465ecede3307230d35/sqlglot/generator.py#L134-L291)


## [Apache DataFusion](https://datafusion.apache.org/)
- General `LogicalPlan` stuff
    - https://www.youtube.com/watch?v=EzZTLiSJnhY
    - https://docs.google.com/presentation/d/1ypylM3-w60kVDW7Q6S99AHzvlBgciTdjsAfqNP85K30
- Setting boundaries for what representations/stages we'll have
    - No optimizer, no physical plan


## [cudf-polars](https://github.com/rapidsai/cudf/tree/d36bfd33ede11d52d19964f89c1512febb62d1df/python/cudf_polars/cudf_polars/dsl)

[^3]: Renaming `ExprIR` -> `Expr` and `NamedIR` -> `NamedExpr`, might be an interesting idea

- Writing non-trivial traversal & translation of the `polars` dsl in python is quite possible
- Separation of [`Expr`](https://github.com/rapidsai/cudf/blob/96896b17420f158d0ce2a024ee0cc24ab712dd6a/python/cudf_polars/cudf_polars/dsl/expressions/base.py#L36-L37) and [`NamedExpr`](https://github.com/rapidsai/cudf/blob/96896b17420f158d0ce2a024ee0cc24ab712dd6a/python/cudf_polars/cudf_polars/dsl/expressions/base.py#L146-L152) [^3]
- Early version of `ExprIR` adapted the inverse of [`Node._non_child`](https://github.com/rapidsai/cudf/blob/96896b17420f158d0ce2a024ee0cc24ab712dd6a/python/cudf_polars/cudf_polars/dsl/nodebase.py#L33-L61) for traversal via slot names
