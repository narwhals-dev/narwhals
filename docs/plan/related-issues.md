# Related issues

## Fully/partially addressed vs `main`
### Expression Expansion
- [x] Support zipped expansion for binary expressions ([#2291])
    - [`ExprIR.iter_expand`][narwhals._plan.expressions.ExprIR.iter_expand] and [`iter_expand_by_combination`][narwhals._plan._nodes.ExprTraverser.iter_expand_by_combination]
- [x] Support broadcast expansion for both sides of binary expressions ([#2244])
    - [`ExprIR.iter_expand`][narwhals._plan.expressions.ExprIR.iter_expand]
- [x] Fix expanding `when` expressions to match Polars ([#3029])
    - [`ExprIR.iter_expand`][narwhals._plan.expressions.ExprIR.iter_expand] and [`iter_expand_by_combination`][narwhals._plan._nodes.ExprTraverser.iter_expand_by_combination]
- [x] Formalise expression expansion in `group_by` ([#2225])
    - [`SelectorIR.iter_expand_selector`][narwhals._plan.expressions.selectors.SelectorIR.iter_expand_selector]
    - [`Ignored`][narwhals._plan.typing.Ignored]
    - [`prepare_projection`][narwhals._plan._expansion.prepare_projection]
    - [`test_group_by_exclude_keys`](https://github.com/narwhals-dev/narwhals/blob/4240b693ff098fb22d5cb3afb72b85c0b01d56b6/tests/plan/group_by_test.py#L725-L760)
    - [`test_group_by_consistent_exclude_21773`](https://github.com/narwhals-dev/narwhals/blob/4240b693ff098fb22d5cb3afb72b85c0b01d56b6/tests/plan/group_by_test.py#L763-L803)
- [x] Unify error message if selector ends up with zero columns ([#2469])
    - `require_any` is accepted by (and the default) for [`expand_selectors`][narwhals._plan._expansion.expand_selectors], 
      [`parse_expand_selectors`][narwhals._plan._expansion.parse_expand_selectors]

### Expressions
- [x] Add `Expr.meta` namespace ([#2869])
    - [`MetaNamespace`][narwhals._plan.expr.MetaNamespace]
- [x] Add `Selector.__xor__` ([#2589])
    - [`Selector.__xor__`][narwhals._plan.selectors.Selector.__xor__]
- [ ] Add `int_range` ([#2722])
    - [x] https://github.com/narwhals-dev/narwhals/issues/2916
    - [ ] https://github.com/narwhals-dev/narwhals/issues/2917
        - [x] https://github.com/narwhals-dev/narwhals/pull/3647
- [ ] Add remaining `Expr.is_*` mask methods ([#3028])
    - [x] [`Expr.is_not_null`][narwhals._plan.expr.Expr.is_not_nan]
    - [x] [`Expr.is_not_nan`][narwhals._plan.expr.Expr.is_not_nan]
    - [ ] `Expr.is_infinite`
- [x] `Expr.is_in(Iterable)` raises inconsistently ([#3195])
    - Validated [at the source] [^1]
    - Introduced specializations 
        - [`IsInExpr`][narwhals._plan.expressions.boolean.IsInExpr]
        - [`IsInSeq`][narwhals._plan.expressions.boolean.IsInSeq] ([guarantees] `tuple`)
        - [`IsInSeries`][narwhals._plan.expressions.boolean.IsInSeries]
- [x] Support `null_count` in eager-only `group_by` ([#2484])
    - Included in the [expanded pyarrow group_by support]
- [ ] Add `unnest` ([#3476])
    - [ ] `Expr.struct.unnest`
    - [x] [`Series.struct.unnest`][narwhals._plan.series.SeriesStructNamespace.unnest]
    - [x] [`DataFrame.unnest`][narwhals._plan.dataframe.DataFrame.unnest]
    - [x] [`LazyFrame.unnest`][narwhals._plan.lazyframe.LazyFrame.unnest]
- [x] Allow for scalars in `sum_horizontal` ([#1868]) [^2]
- [x] Add `is_elementwise` and `returns_scalar` to `map_batches` ([#2522])
    - [`Expr.map_batches`][narwhals._plan.expr.Expr.map_batches]
- [x] Support `descending` and `nulls_last` in `over` ([#2790])
    - [`Expr.over`][narwhals._plan.expr.Expr.over]
    - Supports `Expr`s in `*partition_by` 
    - Supports `Selector`s in `order_by`
- [x] Add `Expr.hist` ([#1561])
    - [`Expr.hist`][narwhals._plan.expr.Expr.hist]
- [ ] Add `Expr.meta.serialize`, `Expr.deserialize` ([#2704]) [^3]
    - [x] `pickle.dumps`
    - [ ] `pickle.loads`
        - *Quick implementation* (https://github.com/narwhals-dev/narwhals/compare/expr-ir/docs/fluff-1...expr-ir/serde-2)
- [ ] Add `Expr.map_elements` ([#3512])
    - Anticipated in [`AnonymousExpr`][narwhals._plan.expressions.function_expr.AnonymousExpr]
    - [`FunctionFlags`][narwhals._plan._flags.FunctionFlags] could be factored out if there are no plans to support

[guarantees]: https://github.com/narwhals-dev/narwhals/blob/4240b693ff098fb22d5cb3afb72b85c0b01d56b6/src/narwhals/_plan/expressions/boolean.py#L60-L63
[expanded pyarrow group_by support]: https://github.com/narwhals-dev/narwhals/blob/4240b693ff098fb22d5cb3afb72b85c0b01d56b6/src/narwhals/_plan/arrow/group_by.py#L64-L108
[at the source]: https://github.com/narwhals-dev/narwhals/blob/4240b693ff098fb22d5cb3afb72b85c0b01d56b6/src/narwhals/_plan/expr.py#L515-L525

### General
- [x] (#3042), (#3059), (#2786)
    - [Plugins] make [`narwhals.Implementation`][] less special
- [x] Add `Series.from_native` ([#2926])
    - [`Series.from_native`][narwhals._plan.series.Series.from_native]
- [x] Add `DataFrame.from_native` ([#2927])
    - [`DataFrame.from_native`][narwhals._plan.dataframe.DataFrame.from_native]
- [x] Add `LazyFrame.from_native` ([#2928])
    - [`LazyFrame.from_native`][narwhals._plan.lazyframe.LazyFrame.from_native]
- [x] Add `pivot` to Arrow ([#2179])
    - https://github.com/narwhals-dev/narwhals/pull/3373


[#2179]: https://github.com/narwhals-dev/narwhals/issues/2179
[#2291]: https://github.com/narwhals-dev/narwhals/issues/2291
[#3029]: https://github.com/narwhals-dev/narwhals/issues/3029
[#3512]: https://github.com/narwhals-dev/narwhals/issues/3512
[#2722]: https://github.com/narwhals-dev/narwhals/issues/2722
[#2704]: https://github.com/narwhals-dev/narwhals/issues/2704
[#3028]: https://github.com/narwhals-dev/narwhals/issues/3028
[#3195]: https://github.com/narwhals-dev/narwhals/issues/3195
[#2926]: https://github.com/narwhals-dev/narwhals/issues/2926
[#2927]: https://github.com/narwhals-dev/narwhals/issues/2927
[#2928]: https://github.com/narwhals-dev/narwhals/issues/2928
[#2869]: https://github.com/narwhals-dev/narwhals/issues/2869
[#2589]: https://github.com/narwhals-dev/narwhals/issues/2589
[#2484]: https://github.com/narwhals-dev/narwhals/issues/2484
[#2225]: https://github.com/narwhals-dev/narwhals/issues/2225
[#2469]: https://github.com/narwhals-dev/narwhals/issues/2469
[#3476]: https://github.com/narwhals-dev/narwhals/issues/3476
[#1868]: https://github.com/narwhals-dev/narwhals/issues/1868
[#2522]: https://github.com/narwhals-dev/narwhals/issues/2522
[#2790]: https://github.com/narwhals-dev/narwhals/issues/2790
[#2244]: https://github.com/narwhals-dev/narwhals/issues/2244
[#1561]: https://github.com/narwhals-dev/narwhals/issues/1561

[^1]: The cool kids call this *"parse, don't validate"*
[^2]: Scalars are always parsed with `lit` (see [`narwhals/_plan/_parse.py`]).
      
      This digs up (#571) again, but ideally that would be an *opt-in* behavior in a future version.
      Where we use a special `ExprIR` node (not `Col` or `Lit`) that unambiguously identifies what is in there - so all
      other backends can reject it.
[^3]: [`Immutable`][narwhals._plan._immutable.Immutable] makes this trivial to support


[`narwhals/_plan/_parse.py`]: https://github.com/narwhals-dev/narwhals/blob/958094160d8a48bf1041ee62d0979bf66b1eec17/src/narwhals/_plan/_parse.py
[Plugins]: ../api-reference-plan/plugins.md#plugins

## Cross-polination
Ideas that have already filtered back in some form

- [ ] https://github.com/narwhals-dev/narwhals/issues/2959
    - Mentioned in [(#3552 comment)](https://github.com/narwhals-dev/narwhals/pull/3552#issuecomment-4270517139)
    - See of experimentation in [tests/plan/conftest.py] and [tests/plan/utils.py]
    - [x] https://github.com/narwhals-dev/narwhals/issues/2872


[tests/plan/conftest.py]: https://github.com/narwhals-dev/narwhals/blob/c57e72c801b5817dc9d20262029fc6c6143c31b2/tests/plan/conftest.py
[tests/plan/utils.py]: https://github.com/narwhals-dev/narwhals/blob/c57e72c801b5817dc9d20262029fc6c6143c31b2/tests/plan/utils.py


## Upstream (`main`) blockers/wants

- [ ] Supertyping #3396 (also unblocks #3386)
    - [`BinaryExpr.resolve_dtype`][narwhals._plan.expressions.BinaryExpr.resolve_dtype]
    - [`TernaryExpr.resolve_dtype`](https://github.com/narwhals-dev/narwhals/blob/4240b693ff098fb22d5cb3afb72b85c0b01d56b6/src/narwhals/_plan/expressions/expr.py#L376-L382)
    - [Several functions](https://github.com/narwhals-dev/narwhals/blob/4240b693ff098fb22d5cb3afb72b85c0b01d56b6/src/narwhals/_plan/expressions/function_expr.py#L178-L189)
    - #3386
         - [Ready to plug in](https://github.com/narwhals-dev/narwhals/blob/4240b693ff098fb22d5cb3afb72b85c0b01d56b6/src/narwhals/_plan/plans/conversion.py#L241-L247)
         - [`VConcatOptions.to_supertypes`][narwhals._plan.options.VConcatOptions.to_supertypes]
- [ ] [`nw.Null`](https://github.com/narwhals-dev/narwhals/issues/2835)
    - Currently overloading [`nw.Unknown`][narwhals.dtypes.Unknown] 
