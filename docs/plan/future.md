# Future work
I've accumulated many, MANY `# TODO @dangotbanned: ...` comments. 
Here's where the more important one's will incubate[^1].


[^1]: and maybe one day, be written up as issues

## `ExprIR` & `Function`
- Remove [`ir.Column` and `ir.Len` aliases](https://github.com/narwhals-dev/narwhals/blob/68fb935cac85b5f69ef0232922611fe9df98c6c9/docs/api-reference-plan/ir/expr-ir/index.md#L3)
- Rename `ExprIR` -> `Expr`, `NamedIR` -> `NamedExpr` ([cudf-polars](./inspired.md#cudf-polars))
- Rename [`RenameAlias` -> `MapAlias`](https://github.com/narwhals-dev/narwhals/blob/68fb935cac85b5f69ef0232922611fe9df98c6c9/src/narwhals/_plan/expressions/name.py#L43-L44)
- Rename [`OverOrdered.sort_options` -> `OverOrdered.options`](https://github.com/narwhals-dev/narwhals/blob/68fb935cac85b5f69ef0232922611fe9df98c6c9/src/narwhals/_plan/expressions/expr.py#L280-L283)
- More ergonomic module/sub-package namespacing
    - [`Function` and friends](https://github.com/narwhals-dev/narwhals/blob/68fb935cac85b5f69ef0232922611fe9df98c6c9/src/narwhals/_plan/expressions/functions.py#L3-L5)
    - `_build` for many things in [misc](../api-reference-plan/ir/misc.md)
- Split and finish [`MetaNamespace` -> (`IRMetaNamespace`, `ExprMetaNamespace`)](https://github.com/narwhals-dev/narwhals/blob/68fb935cac85b5f69ef0232922611fe9df98c6c9/src/narwhals/_plan/meta.py#L124-L141)
- Opening up [`_parse.py`](https://github.com/narwhals-dev/narwhals/blob/68fb935cac85b5f69ef0232922611fe9df98c6c9/src/narwhals/_plan/_parse.py) for extension
    - **PandasLike pre-requisite** ([because why use names for column names?](https://github.com/narwhals-dev/narwhals/blob/68fb935cac85b5f69ef0232922611fe9df98c6c9/docs/plan/related-issues.md?plain=1#L105-L109))
- Replace [`Aggregation`][narwhals._plan._function.Aggregation] functions with [`AggExpr`][narwhals._plan.expressions.aggregation.AggExpr]
    - [Proposal](https://github.com/narwhals-dev/narwhals/blob/68fb935cac85b5f69ef0232922611fe9df98c6c9/src/narwhals/_plan/expressions/aggregation.py#L4-L18)
- Explore visitor & transformer protocols/classes (unrelated to dispatch/compliant)
    - [narwhals/_plan/_rewrites.py](https://github.com/narwhals-dev/narwhals/blob/68fb935cac85b5f69ef0232922611fe9df98c6c9/src/narwhals/_plan/_rewrites.py)
    - [narwhals/_plan/altair/api/expression.py](https://github.com/narwhals-dev/narwhals/blob/0a614b4f2f04cdadcc9e6a7bf177ba87945ca546/src/narwhals/_plan/altair/api/expression.py)
    - [narwhals/_plan/altair/api/aggregate.py](https://github.com/narwhals-dev/narwhals/blob/0a614b4f2f04cdadcc9e6a7bf177ba87945ca546/src/narwhals/_plan/altair/api/aggregate.py)
- Explore `ExtensionFunction` support ([hinted at](https://github.com/narwhals-dev/narwhals/blob/68fb935cac85b5f69ef0232922611fe9df98c6c9/src/narwhals/_plan/expressions/function_expr.py#L300-L305))
    - Main concerns: managing a registry; support user-implementation *for* builtin backends; minimizing the number of dances you need to do

## Compliant
[more flexible signatures for CompliantColumn methods]: https://github.com/narwhals-dev/narwhals/blob/68fb935cac85b5f69ef0232922611fe9df98c6c9/docs/plan/whats-new/behind-the-scenes.md?plain=1#L36-L38

- Finish migrating to post-`CompliantFrame`/`CompliantNamespace` typing [1](https://github.com/narwhals-dev/narwhals/blob/68fb935cac85b5f69ef0232922611fe9df98c6c9/src/narwhals/_plan/compliant/typing.py#L225-L229), 
  [2](https://github.com/narwhals-dev/narwhals/blob/68fb935cac85b5f69ef0232922611fe9df98c6c9/src/narwhals/_plan/compliant/expr.py#L276-L277)
- Use [more flexible signatures for CompliantColumn methods]
    - Related function registry builder idea in `expr-ir/spiraling` stash
- Finish removing boilerplate from (the non-plugin parts of) `CompliantClasses`
    - [narwhals/_plan/arrow/classes.py](https://github.com/narwhals-dev/narwhals/blob/68fb935cac85b5f69ef0232922611fe9df98c6c9/src/narwhals/_plan/arrow/classes.py)
    - [narwhals/_plan/arrow/v1.py](https://github.com/narwhals-dev/narwhals/blob/68fb935cac85b5f69ef0232922611fe9df98c6c9/src/narwhals/_plan/arrow/v1.py)
    - [narwhals/_plan/arrow/v2.py](https://github.com/narwhals-dev/narwhals/blob/68fb935cac85b5f69ef0232922611fe9df98c6c9/src/narwhals/_plan/arrow/v2.py)
    - Related (dynamic & stubs) idea in `expr-ir-fancy-dunder-narwhals-classes` stash
    - Maybe utilizing [PEP 810](https://peps.python.org/pep-0810/) when available

## Backends
### Arrow
- Convert parts of [`arrow.functions`](https://github.com/narwhals-dev/narwhals/blob/68fb935cac85b5f69ef0232922611fe9df98c6c9/src/narwhals/_plan/arrow/functions/__init__.py#L1-L2) into stubs
    - Leaning on [arrow.functions.meta](https://github.com/narwhals-dev/narwhals/blob/68fb935cac85b5f69ef0232922611fe9df98c6c9/src/narwhals/_plan/arrow/functions/meta.py) for runtime

### Polars
- [Shrink `PolarsExpr`](https://github.com/narwhals-dev/narwhals/blob/68fb935cac85b5f69ef0232922611fe9df98c6c9/src/narwhals/_plan/polars/expr.py#L47-L51)
    - E.g. like Arrow's [`unary`,`unary_accessor`](https://github.com/narwhals-dev/narwhals/blob/68fb935cac85b5f69ef0232922611fe9df98c6c9/src/narwhals/_plan/arrow/expr.py#L153-L294) or [more flexible signatures for CompliantColumn methods]

## Lazy
- [Remove `BaseFrame`](https://github.com/narwhals-dev/narwhals/blob/68fb935cac85b5f69ef0232922611fe9df98c6c9/src/narwhals/_plan/dataframe.py#L77-L81)
- [Remove `LazyFrame._compliant`](https://github.com/narwhals-dev/narwhals/blob/68fb935cac85b5f69ef0232922611fe9df98c6c9/src/narwhals/_plan/lazyframe.py#L54-L63)
- [Implement `LazyFrame.to_native`](https://github.com/narwhals-dev/narwhals/blob/68fb935cac85b5f69ef0232922611fe9df98c6c9/src/narwhals/_plan/lazyframe.py#L63)
- Clean up [`Resolver.group_by`](https://github.com/narwhals-dev/narwhals/blob/68fb935cac85b5f69ef0232922611fe9df98c6c9/src/narwhals/_plan/plans/conversion.py#L288-L356)
- Actually solve [`Scan*` metadata](https://github.com/narwhals-dev/narwhals/blob/68fb935cac85b5f69ef0232922611fe9df98c6c9/src/narwhals/_plan/plans/conversion.py#L678-L707) ([see also](https://github.com/narwhals-dev/narwhals/blob/68fb935cac85b5f69ef0232922611fe9df98c6c9/src/narwhals/_plan/arrow/io.py#L126-L132))
- Reuse the [`Node` protocol](https://github.com/narwhals-dev/narwhals/blob/68fb935cac85b5f69ef0232922611fe9df98c6c9/src/narwhals/_plan/_nodes.py#L184-L234) for [`*Plan`](https://github.com/narwhals-dev/narwhals/blob/68fb935cac85b5f69ef0232922611fe9df98c6c9/src/narwhals/_plan/plans/_base.py)
- Typing `Native` 
    - [`Scan[Native]` -> `LazyFrame[Native]`](https://github.com/narwhals-dev/narwhals/blob/68fb935cac85b5f69ef0232922611fe9df98c6c9/src/narwhals/_plan/plans/logical.py#L447-L453)
    - [`LazyFrame.from_*`](https://github.com/narwhals-dev/narwhals/blob/68fb935cac85b5f69ef0232922611fe9df98c6c9/src/narwhals/_plan/lazyframe.py#L74-L75)
    - [`Native` through all `LogicalPlan[Native]`](https://github.com/narwhals-dev/narwhals/blob/68fb935cac85b5f69ef0232922611fe9df98c6c9/src/narwhals/_plan/plans/logical.py#L112)
    - [`LogicalPlan[Native]` -> `LogicalToResolved[Native]`](https://github.com/narwhals-dev/narwhals/blob/68fb935cac85b5f69ef0232922611fe9df98c6c9/src/narwhals/_plan/plans/logical.py#L175-L187)
    - [`LogicalToResolved[Native]` -> `ResolvedPlan[Native]`](https://github.com/narwhals-dev/narwhals/blob/68fb935cac85b5f69ef0232922611fe9df98c6c9/src/narwhals/_plan/plans/visitors.py#L58-L62)
    - [`ResolvedPlan`](https://github.com/narwhals-dev/narwhals/blob/68fb935cac85b5f69ef0232922611fe9df98c6c9/src/narwhals/_plan/plans/resolved.py#L218-L226)

## Misc
- Use `frozendict` ([PEP 814](https://peps.python.org/pep-0814/)) in [`FrozenSchema`](https://github.com/narwhals-dev/narwhals/blob/68fb935cac85b5f69ef0232922611fe9df98c6c9/src/narwhals/_plan/schema.py#L424-L427) when available
