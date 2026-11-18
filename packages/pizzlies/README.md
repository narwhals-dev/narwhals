# pizzlies

A pure-Python, zero-dependency, eager dataframe library whose native frame is a plain
`dict[str, list]`, with the [Narwhals](https://github.com/narwhals-dev/narwhals) API.

```console
pip install pizzlies
```

## Standalone

`pizzlies` re-exports Narwhals' API, with its constructors bound to this backend:

```python
import pizzlies as pz

df = pz.from_dict({"a": [1, 2, 3], "b": [4.0, None, 6.0]})
df.with_columns(c=pz.col("a") * 2).select(pz.col("c").sum(), pz.col("b").mean())
df.to_native()  # {'a': [1, 2, 3], 'b': [4.0, None, 6.0]}
```

`pz.from_dict`, `pz.from_dicts`, `pz.from_numpy`, `pz.from_arrow`, `pz.new_series`,
`pz.read_csv`, `pz.scan_csv`, `pz.read_parquet` and `pz.scan_parquet` create `pizzlies`-backed
objects. Everything else (`pz.col`,
`pz.when`, `pz.concat`, `pz.selectors`, the dtypes, ...) is Narwhals' own, so see the
[Narwhals docs](https://narwhals-dev.github.io/narwhals/) for the full API.

## Lazy frames

`df.lazy()`, `pz.scan_parquet` and `pz.scan_csv` return lazy frames, which record
operations in a query plan instead of running them. `collect()` first prunes the
columns that nothing reads (projection pushdown), so that `scan_parquet` only decodes
the columns a query uses, and then runs the plan eagerly:

```python
lf = pz.scan_parquet("lineitem.parquet")  # reads only the footer
result = lf.filter(pz.col("l_quantity") > 45).select(pz.col("l_extendedprice").sum())
print(result.to_native().explain())  # the plan, as a tree
result.collect()  # reads `l_quantity` and `l_extendedprice` only
```

Operations are checked when they're defined, by running them on empty frames, so e.g.
selecting a missing column raises straight away. Some errors which depend on dtypes
(which `pizzlies` infers from values) only show up on `collect()`. Expressions whose
columns depend on the input's dtypes or order (selectors, `all()`, `nth`, ...) keep
every column, and resolving dtype selectors runs the plan to get its schema.

## As a Narwhals plugin

Narwhals discovers the plugin automatically, so a dict of lists can be passed straight to
`nw.from_native`, and `backend="pizzlies"` (or `backend=pz.BACKEND`) works wherever Narwhals
accepts a `backend`:

```python
import narwhals as nw

df = nw.from_native({"a": [1, 2, 3]}, eager_only=True)
df.select(nw.col("a").sum()).to_native()  # {'a': [6]}
```

Note that, once installed, `pizzlies` claims every plain `dict` passed to `nw.from_native`.

## Limitations

Parquet files are read by a small pure-Python reader, straight into a dict of lists. It
supports flat columns; data pages V1 and V2; `PLAIN`, dictionary and boolean `RLE` encodings;
and uncompressed, snappy, gzip and zstd compression. Zstd (Polars' default) uses the
stdlib's `compression.zstd` on Python 3.14+, and on older Pythons needs the optional
`backports.zstd` package: `pip install pizzlies[zstd]`. Anything else (nested columns, `DELTA_*`
encodings, brotli/lz4, ...) raises `NotImplementedError`: read such files with pyarrow and
convert them, `pz.from_arrow(pq.read_table(path))`. Writing Parquet isn't supported.
`Map` dtypes aren't supported, and dtypes are inferred
from the Python values, so e.g. integers are always `Int64`.

## Development

The public API lives in `pizzlies/__init__.py`, and the Narwhals-compliant implementation in
the private `pizzlies._plugin`. It only imports from Narwhals' public API
(`narwhals.compliant`, `narwhals.typing`, `narwhals.utils`, ...): anything it needs from
Narwhals' private modules is vendored in `pizzlies/_plugin/_vendored.py`. The exception is
expression metadata (`Expr._metadata`), which the plugin reads directly and which has no
stability guarantees, so the plugin needs to be kept in step with the Narwhals versions it
supports.

The package has its own environment and lock file, and is not part of the root uv workspace:
once installed, it claims every plain `dict`, which would clash with the in-repo
`test-plugin`.

`run_tests.py` runs, in this package's `.venv`, both this package's own tests and Narwhals'
test suite against the plugin. Narwhals tests which are known to fail (mostly
not-yet-implemented methods) are listed in `known_failures.txt` and deselected:

```console
python packages/pizzlies/run_tests.py           # excludes known failures
python packages/pizzlies/run_tests.py --all     # includes known failures
python packages/pizzlies/run_tests.py --update  # regenerate known_failures.txt
```

`--update` exits non-zero if any test was added to the baseline, so new failures can't be
absorbed into it unnoticed.
