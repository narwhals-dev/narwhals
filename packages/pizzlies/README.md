# pizzlies

A pure-Python dataframe library, whose only dependency is
[Narwhals](https://github.com/narwhals-dev/narwhals), with Narwhals' API. Its eager frames
are plain `dict[str, list]`s, and it has lazy frames with projection pushdown.

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
`pz.read_csv`, `pz.scan_csv`, `pz.read_parquet` and `pz.scan_parquet` create
`pizzlies`-backed objects. Everything else (`pz.col`, `pz.when`, `pz.concat`,
`pz.selectors`, the dtypes, ...) is Narwhals' own, so see the
[Narwhals docs](https://narwhals-dev.github.io/narwhals/) for the full API.

## Lazy frames

`df.lazy()`, `pz.scan_parquet` and `pz.scan_csv` return lazy frames, which record
operations in a query plan instead of running them (`scan_csv` does parse the whole file
straight away). `collect()` first works out which columns the query reads (projection
pushdown), so that `scan_parquet` only decodes those, and then runs the plan eagerly:

```python
lf = pz.scan_parquet("lineitem.parquet")  # reads only the footer
result = lf.filter(pz.col("l_quantity") > 45).select(pz.col("l_extendedprice").sum())
print(result.to_native().explain())  # the plan, as a tree
result.collect()  # reads `l_quantity` and `l_extendedprice` only
```

There are no other optimisations, and `collect()` runs the plan every time. Errors in
the query are raised on `collect()`, as in Polars: errors which only depend on column
names (e.g. a missing or duplicate column) before anything is read, and others while the
plan runs. Some are raised straight away though: `scan_parquet` reads the file's footer
(and checks `columns` against it), `scan_csv` parses the whole file, and some Narwhals
methods check column names, e.g. `cast`, so name errors can also come from those. So,
unlike in eager mode, a query with both kinds of error raises the name error. Dtypes are
inferred from values, so `collect_schema()` runs the plan (once per frame), and so do
dtype selectors, to resolve which columns they select; `columns` doesn't run anything
otherwise. `explain()` shows a subplan used more than once only once, and refers to it by
number after that.

Inputs keep every column whenever it's unclear which ones an expression reads (for
selectors, `all()`, `nth` and `exclude`), and also for `unique()` and `drop_nulls()`
without a subset, diagonal and horizontal `concat`, and `unpivot` without `on`.

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

## Size, import time and performance

`pizzlies` is small and quick to import, but it's pure Python, so it's much slower than
Polars on anything but small data.

| | Installed size | Import time |
| --- | ---: | ---: |
| `pizzlies` (+ `narwhals`) | 0.3 MB (+ 2.1 MB) | 40 ms |
| `polars` (+ `polars-runtime-32`) | 183 MB | 125 ms |
| `pyarrow` | 171 MB | 144 ms |
| `pandas` (+ `numpy`) | 96 MB | 365 ms |

The `pizzlies` wheel is 74 KB. Import times are the median of 40 fresh interpreters, from
wheel installs; almost all of `pizzlies`' is importing Narwhals (34 ms on its own).

On the 22 [TPC-H](https://github.com/narwhals-dev/narwhals/tree/main/tpch) queries at
scale factor 0.1 (`lineitem` has 600,000 rows), run through Narwhals end to end, including
reading the Parquet files (`scan_parquet`, query, `collect` for lazy frames; `read_parquet`
and query for eager ones):

| | Total | Slowdown vs Polars lazy (geometric mean over queries) |
| --- | ---: | ---: |
| `polars` lazy | 0.76 s | 1x |
| `polars` eager | 2.95 s | 3.7x |
| `pizzlies` lazy | 10.6 s | 11.9x |
| `pizzlies` eager | 27.5 s | 31.2x |

Most of the gap between `pizzlies`' lazy and eager frames is reading: lazy frames only
decode the columns a query uses.

Measured on Linux x86-64 (AMD Ryzen AI 9 465), with Python 3.13 for import times and 3.14
for TPC-H, and pandas 3.1, Polars 2.1 and PyArrow 26. Results depend on your hardware and data, so measure your own workload.
`benchmark.py` reproduces the import times and TPC-H numbers, see its docstring.

## Limitations

`pizzlies` may be stricter than Polars, but it's never silently wrong: where it doesn't
do what Polars does, it raises. For example, `unpivot` of columns with different dtypes
raises, where Polars casts them to a common dtype (integers with floats are fine). The
exceptions, where `pizzlies` follows Python instead, are listed below.

- **Parquet:** `pz.read_parquet` uses a small pure-Python reader. It handles flat and
  nested (list, struct and map) columns, the `PLAIN`, dictionary, `RLE`,
  `BYTE_STREAM_SPLIT` and `DELTA_*` encodings, and uncompressed, snappy, gzip, LZ4 and
  zstd compression. Zstd, Polars' default, needs Python 3.14+ or
  `pip install 'pizzlies[zstd]'`. Other files (e.g. brotli-compressed ones) raise
  `NotImplementedError`: read them with pyarrow and convert them,
  `pz.from_arrow(pq.read_table(path))`. Corrupt data raises `ValueError`, and page
  checksums are verified when the writer stored them, though only for the columns read.
- **Dtypes** are inferred from the Python values, so e.g. integers are always `Int64`,
  and an empty or all-null column's dtype is `Unknown` (so dtype selectors don't select
  it). `Map` dtypes aren't supported, so maps are read as lists of
  `{"key": ..., "value": ...}` structs, as Polars does.
- **NaN:** comparisons follow IEEE 754, as in Python: e.g. `NaN == NaN` and `NaN > 1.0`
  are false, where Polars treats NaN as equal to itself and greater than any other
  value. Sorting, `unique`, `group_by` and joins do treat NaN as Polars does.
- **Floats** follow Python in a few details: a `Float64` column can hold integers (e.g.
  after `fill_null(1.5)` on an integer column), NaN casts to the string `"nan"` (Polars:
  `"NaN"`), and `-5.0 % math.inf` is `inf` (Polars: NaN).
- **No columns, no rows:** a frame is a dict of lists, so one with no columns has no
  rows either, e.g. `df.drop(*df.columns).with_columns(a=pz.lit(1))` is empty.
- **Data isn't copied,** as with Narwhals and any backend: `nw.from_native(data)` wraps
  your dict as it is, so later changes to it show up in the frame, and results share the
  lists of columns they leave unchanged with their inputs. `.lazy()` copies which
  columns there are, but not their lists. Don't modify the lists of a frame (or of the
  dict it was built from) in place.
- **Time zones:** like Python, and unlike Polars, `pizzlies` ignores `fold` when comparing
  time-zone aware datetimes, so the two occurrences of an ambiguous wall time (e.g. 01:30
  when clocks go back) are equal in group-bys, joins and `unique`.

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
