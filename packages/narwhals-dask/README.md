# narwhals-dask

[Dask](https://www.dask.org/) backend for [Narwhals](https://github.com/narwhals-dev/narwhals).

Dask support is moving out of Narwhals and into this package. Narwhals still ships a
built-in Dask backend for now, but emits a `FutureWarning` when it's used. Installing
`narwhals-dask` silences that warning, and will be required in a future release:

```console
pip install narwhals-dask
```

There's nothing else to do: Narwhals discovers the plugin automatically, and
`nw.from_native(dask_df)`, `backend="dask"` and `backend="narwhals-dask"` all use it.

```python
import dask.dataframe as dd
import narwhals as nw

df = nw.from_native(dd.from_dict({"a": [1, 2, 3]}, npartitions=1))
df.select(nw.col("a").sum()).collect()
```

## Development

`narwhals-dask` never imports from Narwhals' private modules: anything it needs from them
is vendored in `narwhals_dask/_vendored.py`. The exception is expression metadata
(`Expr._metadata`, e.g. `op_nodes_reversed()`, `expansion_kind`, `is_literal`), which the
plugin reads directly and which has no stability guarantees, so the plugin needs to be kept in
step with the Narwhals versions it supports.

Narwhals' own test suite exercises the plugin once it's installed. From this directory:

```console
uv run pytest ../../tests --constructors=dask
```
