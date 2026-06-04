# API Reference (`narwhals._plan`)

!!! warning
    This namespace is highly experimental and incomplete.
    You should expect to see both new and missing features - but nothing
    is planned to be removed if it exists on `main`.

- Narwhals-level
    - [Functions](functions.md)
    - [DataFrame](dataframe.md)
        - GroupBy
    - [LazyFrame](lazyframe.md)
        - GroupBy
    - [Expressions](expr/index.md)
        - [Categorical](expr/categorical.md)
        - [List](expr/list.md)
        - [Meta](expr/meta.md)
        - [Name](expr/name.md)
        - [String](expr/string.md)
        - [Struct](expr/struct.md)
        - [Temporal](expr/temporal.md)
    - [Selectors](selectors.md)
    - Series
- [IR](ir/index.md)
    - Logical
        - [ExprIR](ir/expr-ir/index.md)
        - SelectorIR
        - LogicalPlan
    - Resolved
        - NamedIR
        - ResolvedPlan
- Compliant-level
    - CompliantClasses
    - CompliantDataFrame
        - GroupBy
    - CompliantLazyFrame
    - CompliantExpr
    - CompliantScalar
    - CompliantSeries