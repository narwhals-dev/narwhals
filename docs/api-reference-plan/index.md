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
    - [Expressions](expressions/index.md)
        - [Categorical](expressions/categorical.md)
        - [List](expressions/list.md)
        - [Meta](expressions/meta.md)
        - [Name](expressions/name.md)
        - [String](expressions/string.md)
        - [Struct](expressions/struct.md)
        - [Temporal](expressions/temporal.md)
    - [Selectors](selectors.md)
    - Series
- IR-level
    - Logical
        - ExprIR
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