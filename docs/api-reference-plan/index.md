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
    - [Expressions](expr.md)
        - [Categorical](expr_cat.md)
        - [List](expr_list.md)
        - Meta
        - [Name](expr_name.md)
        - [String](expr_str.md)
        - [Struct](expr_struct.md)
        - [Temporal](expr_dt.md)
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