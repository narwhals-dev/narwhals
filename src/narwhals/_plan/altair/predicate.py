"""Translation of `ExprIR` into a Vega-Lite [Predicate].

They're part of the requirements for `conditional.py` and are not very intuitive to write in `altair` (see [vega/altair/#3239]).

[Predicate]: https://vega.github.io/vega-lite/docs/predicate.html
[vega/altair/#3239]: https://github.com/vega/altair/issues/3239

## [`FieldPredicate`](https://vega.github.io/vega-lite/docs/predicate.html#field-predicate)

They look like:

    class FieldSomethingPredicate:
        something: Any
        field: str
        timeUnit: MultiTimeUnit_T | BinnedTimeUnit_T | SingleTimeUnit_T


- Binary comparisons (all simple, but see `ExprRef`)
    - `FieldEqualPredicate`
    - `FieldGTEPredicate`
    - `FieldGTPredicate`
    - `FieldLTEPredicate`
    - `FieldLTPredicate`
- `FieldOneOfPredicate`
    - (`col("a").is_in([1, 3, 5]))`) -> (`field="a", oneOf=[1, 3, 5]`)
- `FieldRangePredicate`
    - (`int_range(1, 5).alias("a")`) -> (`field="a", range=[1, 4]`)
        - polars (inclusive, exclusive)
        - vega (inclusive, inclusive)
            - like `closed: ClosedInterval = "both"`
        - Could also support (`int_range(...).alias("a").cast(Float{32,64})`)
            - `linear_space` is a float range, but seems too complex for this task
    - (`date_range(start, end, interval, closed="both").alias("a")`) ->
        - (`field="a", range=[start, end], timeUnit=translate(interval)`)
        - Would need to lift the current `narwhals._plan` restriction on `interval` to allow the full range of `timeUnit`
            - e.g. parse into `Interval` but don't raise early when we can't convert to days
            - add this to as a *Roadmap* task, since backends besides `pyarrow` can handle more
- `FieldValidPredicate`
    - (`col("a").is_not_null() & col("a").is_not_nan()`) -> (`field="a", valid=True`)
    - (`col("a").is_null() & col("a").is_nan()`) -> (`field="a", valid=False`)

## [`PredicateComposition`](https://vega.github.io/vega-lite/docs/predicate.html#composition)

- BinaryExpr
    - (`ops.And`) -> `LogicalAndPredicate({"and": [left, right]})`
    - (`ops.Or`) -> `LogicalOrPredicate({"or": [left, right]})`
    - (`ops.ExclusiveOr`) ->
        - `PredicateComposition({"or": [{"and": [left, {"not": right}]}, {"and": [{"not": left}, right]}]})`
- FunctionExpr
    - (`F.Not`)` -> `LogicalNotPredicate({"not": expr.args[0]})`
    - Need to special-case (`ops.NotEq`)

## [`ExprRef`]

[`ExprRef`]: https://vega.github.io/vega-lite/docs/types.html#exprref

Binary comparisons can take an [`ExprRef`] instead of just primitive values:

    FieldEqualPredicate(field="a", equal=1)
    FieldEqualPredicate(field="a", equal={"expr": "abs(datum.b)"})


### Important
If we encounter something that isn't a [Predicate], we then need to check if `expression.py` can handle it.
"""

from __future__ import annotations
