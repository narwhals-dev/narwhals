# narwhals-dict

A pure-Python, zero-dependency eager backend for
[Narwhals](https://github.com/narwhals-dev/narwhals) whose native frame is a plain
`dict[str, list]`.

```console
pip install narwhals-dict
```

Narwhals discovers the plugin automatically, so a dict of lists can be passed straight to
`nw.from_native`, and `backend="narwhals-dict"` works wherever Narwhals accepts a `backend`:

```python
import narwhals as nw

df = nw.from_native({"a": [1, 2, 3], "b": [4.0, None, 6.0]}, eager_only=True)
df.select(nw.col("a").sum(), nw.col("b").mean()).to_native()
# {'a': [6], 'b': [5.0]}
```

Parquet isn't supported: `nw.read_parquet` and `nw.scan_parquet` raise. Read the file with
pyarrow and convert it instead:

```python
import pyarrow.parquet as pq

df = nw.from_arrow(pq.read_table(path), backend="narwhals-dict")
```

## Development

`narwhals-dict` only imports from Narwhals' public API (`narwhals.compliant`,
`narwhals.typing`, `narwhals.utils`, ...): anything it needs from Narwhals' private modules
is vendored in `narwhals_dict/_vendored.py`. The exception is expression metadata
(`Expr._metadata`), which the plugin reads directly and which has no stability guarantees, so
the plugin needs to be kept in step with the Narwhals versions it supports.

The package has its own environment and lock file, and is not part of the root uv workspace:
once installed, it claims every plain `dict`, which would clash with the in-repo
`test-plugin`.

Narwhals' own test suite runs against the plugin via `run_tests.py`, which uses this
package's `.venv`. Tests which are known to fail (mostly not-yet-implemented methods) are
listed in `known_failures.txt` and deselected:

```console
python packages/narwhals-dict/run_tests.py           # excludes known failures
python packages/narwhals-dict/run_tests.py --all     # includes known failures
python packages/narwhals-dict/run_tests.py --update  # regenerate known_failures.txt
```

`--update` exits non-zero if any test was added to the baseline, so new failures can't be
absorbed into it unnoticed.
