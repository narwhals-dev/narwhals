# Mind the (feature) gap
This is a non-exhaustive list of known feature gaps between `narwhals._plan` and `narwhals`.

See anything you're interested in?  
**Contributions are more than welcome**


## Compliant-level
### `CompliantExpr`
- [ ] [`Expr.str.to_date`][narwhals._plan.expr.ExprStringNamespace.to_date] [^1]
- [ ] [`Expr.str.to_datetime`][narwhals._plan.expr.ExprStringNamespace.to_datetime] [^1]
- [ ] [`Expr.dt.*`][narwhals._plan.expr.ExprDateTimeNamespace] [^1] [^2]
- [ ] Newly expressified method arguments [^3]

### `CompliantLazyFrame`

- [ ] `filter`
- [ ] `with_columns`
- [ ] `group_by` [^8]
- [ ] `group_by_names` [^8]

## Narwhals-level
### [`Expr`][narwhals._plan.expr.Expr]
- [x] feat(expr-ir): Add `list.*` aggregate methods (#3353)
- [x] feat(expr-ir): Add `list.sort` (#3359)
- [x] feat(expr-ir): Add top-level `struct` function (#3666)
- [ ] feat: Add `{Expr,Series}.any_value` (#3315) *
    - **Alternative**: Add `{Expr,Series}.{first,last}(ignore_nulls=...)` (https://github.com/pola-rs/polars/pull/25105)
- [ ] feat: Add `{Expr,Series}.sin` (#3365)
- [ ] feat: Add `{Expr,Series}.cos` (#3392)
- [ ] feat: Add `str.pad_{start,end}` (#3395)
- [ ] feat: Add `nw.corr` (#3460)
- [ ] feat: Add `{Expr,Series}.str.to_time` (#3538)
- [ ] feat: Add `{Expr,Series}.__neg__` (#3625)
- [ ] feat: Add `{Expr,Series}.is_close` (#2962)

### [`DataFrame`][narwhals._plan.dataframe.DataFrame]
- [ ] `head` [^4]
- [ ] `tail` [^4]
- [ ] `estimated_size`
- [ ] `null_count`
- [ ] `rows`
- [ ] `is_duplicated`
- [ ] `is_empty`
- [ ] `is_unique`
- [ ] `item`
- [ ] `iter_rows`
- [ ] `from_arrow` [^5]
- [ ] `from_dicts`
- [ ] `from_numpy`
- [ ] `to_numpy`
- [ ] `top_k`
- [ ] `pipe`
- [ ] `__arrow_c_stream__`
- [ ] `__getitem__` [^6]

### [`LazyFrame`][narwhals._plan.lazyframe.LazyFrame]
- [ ] `top_k`
- [ ] `gather_every` (stable.v1)
- [ ] `lazy`
- [ ] `pipe`
- [ ] `to_native` [^7]


### [`Series`][narwhals._plan.series.Series]
- Mostly not implemented 

[^1]: Available at narwhals-level
[^2]: One day I hope this'll lure @MarcoGorelli in
[^3]: but apparently I wasn't paying close enough attention 🤦‍♂️
[^4]: Don't add to compliant-level, use [`DataFrame.slice`][narwhals._plan.dataframe.DataFrame.slice] instead
[^5]: Available at compliant-level
[^6]: Don't add to compliant-level, `ArrowDataFrame` has `gather` - what other methods do we need to implement this universally?
[^7]: Needs some design work first
[^8]: Low priority, [`Resolver.group_by`](https://github.com/narwhals-dev/narwhals/blob/a5cddb86d092cf2163859d8106da636671322c7f/src/narwhals/_plan/plans/conversion.py#L288-L356) 
      needs to be cleaned up and [`Resolver`](https://github.com/narwhals-dev/narwhals/blob/a5cddb86d092cf2163859d8106da636671322c7f/src/narwhals/_plan/plans/conversion.py#L167-L178) made extensible
