# Related issues

## Fully/partially addressed vs `main`
- [x] https://github.com/narwhals-dev/narwhals/issues/2179
    - https://github.com/narwhals-dev/narwhals/pull/3373
- [x] https://github.com/narwhals-dev/narwhals/issues/2291
    - [`ExprIR.iter_expand`][narwhals._plan.expressions.ExprIR.iter_expand] and [`iter_expand_by_combination`][narwhals._plan._nodes.ExprTraverser.iter_expand_by_combination]
- [x] https://github.com/narwhals-dev/narwhals/issues/3029
    - [`ExprIR.iter_expand`][narwhals._plan.expressions.ExprIR.iter_expand] and [`iter_expand_by_combination`][narwhals._plan._nodes.ExprTraverser.iter_expand_by_combination]
- [ ] https://github.com/narwhals-dev/narwhals/issues/3512
    - Anticipated in [`AnonymousExpr`][narwhals._plan.expressions.function_expr.AnonymousExpr]
    - [`FunctionFlags`][narwhals._plan._flags.FunctionFlags] could be factored out if there are no plans to support
- [ ] https://github.com/narwhals-dev/narwhals/issues/2722
    - [x] https://github.com/narwhals-dev/narwhals/issues/2916
    - [ ] https://github.com/narwhals-dev/narwhals/issues/2917
        - [x] https://github.com/narwhals-dev/narwhals/pull/3647
- [ ] https://github.com/narwhals-dev/narwhals/issues/2704 [^1]
    - [x] `pickle.dumps`
    - [ ] `pickle.loads`
        - Needs a custom [`__setstate__`](https://docs.python.org/3/library/pickle.html#object.__setstate__)
        - *Quick implementation* (https://github.com/narwhals-dev/narwhals/compare/expr-ir/docs/fluff-1...expr-ir/serde-2)
- [ ] https://github.com/narwhals-dev/narwhals/issues/3028
    - [x] [`Expr.is_not_null`][narwhals._plan.expr.Expr.is_not_nan]
    - [x] [`Expr.is_not_nan`][narwhals._plan.expr.Expr.is_not_nan]
    - [ ] `Expr.is_infinite`
- [x] https://github.com/narwhals-dev/narwhals/issues/3195
    - Validated [at the source](https://github.com/narwhals-dev/narwhals/blob/4240b693ff098fb22d5cb3afb72b85c0b01d56b6/src/narwhals/_plan/expr.py#L515-L525)
    - Introduced specializations 
        - [`IsInExpr`][narwhals._plan.expressions.boolean.IsInExpr]
        - [`IsInSeq`][narwhals._plan.expressions.boolean.IsInSeq] ([guarantees](https://github.com/narwhals-dev/narwhals/blob/4240b693ff098fb22d5cb3afb72b85c0b01d56b6/src/narwhals/_plan/expressions/boolean.py#L60-L63) `tuple`)
        - [`IsInSeries`][narwhals._plan.expressions.boolean.IsInSeries]
- [x] (#3042), (#3059), (#2786)
    - [Plugins] make [`narwhals.Implementation`][] less special
- [x] https://github.com/narwhals-dev/narwhals/issues/2926
    - [`Series.from_native`][narwhals._plan.series.Series.from_native]
- [x] https://github.com/narwhals-dev/narwhals/issues/2927
    - [`DataFrame.from_native`][narwhals._plan.dataframe.DataFrame.from_native]
- [x] https://github.com/narwhals-dev/narwhals/issues/2928
    - [`LazyFrame.from_native`][narwhals._plan.lazyframe.LazyFrame.from_native]
- [x] https://github.com/narwhals-dev/narwhals/issues/2869
    - [`MetaNamespace`][narwhals._plan.expr.MetaNamespace]
- [x] https://github.com/narwhals-dev/narwhals/issues/2589
    - [`Selector.__xor__`][narwhals._plan.selectors.Selector.__xor__]
- [x] https://github.com/narwhals-dev/narwhals/issues/2484
    - Included in the [expanded pyarrow group_by support](https://github.com/narwhals-dev/narwhals/blob/4240b693ff098fb22d5cb3afb72b85c0b01d56b6/src/narwhals/_plan/arrow/group_by.py#L64-L108)
- [x] https://github.com/narwhals-dev/narwhals/issues/2225
    - [`SelectorIR.iter_expand_selector`][narwhals._plan.expressions.selectors.SelectorIR.iter_expand_selector]
    - [`Ignored`][narwhals._plan.typing.Ignored]
    - [`prepare_projection`][narwhals._plan._expansion.prepare_projection]
    - [`test_group_by_exclude_keys`](https://github.com/narwhals-dev/narwhals/blob/4240b693ff098fb22d5cb3afb72b85c0b01d56b6/tests/plan/group_by_test.py#L725-L760)
    - [`test_group_by_consistent_exclude_21773`](https://github.com/narwhals-dev/narwhals/blob/4240b693ff098fb22d5cb3afb72b85c0b01d56b6/tests/plan/group_by_test.py#L763-L803)
- [x] https://github.com/narwhals-dev/narwhals/issues/2469
    - `require_any` is accepted by (and the default) for [`expand_selectors`][narwhals._plan._expansion.expand_selectors], [`parse_expand_selectors`][narwhals._plan._expansion.parse_expand_selectors]

[^1]: [`Immutable`][narwhals._plan._immutable.Immutable] makes this trivial to support


[Plugins]: ../api-reference-plan/plugins.md#plugins

## Cross-polination
Ideas that have already filtered back in some form

- [ ] https://github.com/narwhals-dev/narwhals/issues/2959
    - Mentioned in [(#3552 comment)](https://github.com/narwhals-dev/narwhals/pull/3552#issuecomment-4270517139)
    - See of experimentation in [tests/plan/conftest.py] and [tests/plan/utils.py]
    - [x] https://github.com/narwhals-dev/narwhals/issues/2872


[tests/plan/conftest.py]: https://github.com/narwhals-dev/narwhals/blob/c57e72c801b5817dc9d20262029fc6c6143c31b2/tests/plan/conftest.py
[tests/plan/utils.py]: https://github.com/narwhals-dev/narwhals/blob/c57e72c801b5817dc9d20262029fc6c6143c31b2/tests/plan/utils.py


## Upstream blockers/wants

- [ ] Supertyping #3396 (also unblocks #3386)
    - [`BinaryExpr.resolve_dtype`][narwhals._plan.expressions.BinaryExpr.resolve_dtype]
    - [`TernaryExpr.resolve_dtype`](https://github.com/narwhals-dev/narwhals/blob/4240b693ff098fb22d5cb3afb72b85c0b01d56b6/src/narwhals/_plan/expressions/expr.py#L376-L382)
    - [Several functions](https://github.com/narwhals-dev/narwhals/blob/4240b693ff098fb22d5cb3afb72b85c0b01d56b6/src/narwhals/_plan/expressions/function_expr.py#L178-L189)
    - #3386
         - [Ready to plug in](https://github.com/narwhals-dev/narwhals/blob/4240b693ff098fb22d5cb3afb72b85c0b01d56b6/src/narwhals/_plan/plans/conversion.py#L241-L247)
         - [`VConcatOptions.to_supertypes`][narwhals._plan.options.VConcatOptions.to_supertypes]
- [ ] [`nw.Null`](https://github.com/narwhals-dev/narwhals/issues/2835)
    - Currently overloading [`nw.Unknown`][narwhals.dtypes.Unknown] 