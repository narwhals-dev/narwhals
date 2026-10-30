# `narwhals.selectors`

The following selectors are all supported. In addition, just like in Polars, the following
set operations are supported:

- set intersection: `&`
- set union: `|`
- set difference: `-`
- complement: `~`

As in Polars, set operations return columns in the frame's order, so
`ncs.boolean() | ncs.numeric()` and `ncs.numeric() | ncs.boolean()` give the same result.

::: narwhals.selectors
    handler: python
    options:
      members:
        - boolean
        - by_dtype
        - categorical
        - datetime
        - enum
        - matches
        - numeric
        - string
      show_root_heading: false
      show_source: false
      show_bases: false
