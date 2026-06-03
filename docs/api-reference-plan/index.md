# API Reference (`narwhals._plan`)

!!! warning
    This namespace is highly experimental and incomplete.
    You should expect to see both new and missing features - but nothing
    is planned to be removed if it exists on `main`.

- Narwhals-level
    - [Functions](functions.md)
    - [DataFrame](dataframe.md)
        - GroupBy
    - LazyFrame
        - GroupBy
    - Expressions
        - Categorical
        - List
        - Meta
        - Name
        - String
        - Struct
        - Temporal
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