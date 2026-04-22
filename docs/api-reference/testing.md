# `narwhals.testing`

## Assertions

::: narwhals.testing
    handler: python
    options:
      show_root_heading: false
      heading_level: 3
      members:
        - assert_frame_equal
        - assert_series_equal

## `pytest` plugin

Narwhals register a pytest plugin that exposes parametrized fixtures with callables
to build native frames from a column-oriented python `dict`.

### Available fixtures

| Fixture | Backends |
|---|---|
| `nw_frame_constructor` | every selected backend (eager + lazy) |
| `nw_eager_constructor` | only eager backends |
| `nw_pandas_like_constructor` | pandas-like backends |

### Pytest options

The backend selection is controlled by the following CLI options:

* `--nw-backends=pandas,polars[lazy],duckdb`: comma-separated list.
    Defaults to [`DEFAULT_BACKENDS`][narwhals.testing.constructors.DEFAULT_BACKENDS]
    intersected with the backends installed in the current environment.
* `--nw-all-backends`: shortcut for "every **CPU** backend that is installed".
* `--use-nw-external-constructor`: Skip narwhals.testing's parametrisation and let
    another plugin provide the `constructor*` fixtures.

Set the `NARWHALS_DEFAULT_BACKENDS` environment variable to override the default
list (useful e.g. when running under `cudf.pandas`).

### Quick start

The plugin auto-loads as soon as you `pip install narwhals`. Just write a test:

```python
from typing import TYPE_CHECKING

import narwhals as nw

if TYPE_CHECKING:
    from narwhals.testing.typing import EagerFrameConstructor, Data


def test_shape(nw_eager_constructor: EagerFrameConstructor) -> None:
    data: Data = {"x": [1, 2, 3]}
    df = nw.from_native(nw_eager_constructor(data), eager_only=True)
    assert df.shape == (3, 1)
```

The fixtures are parametrised against every supported backend that is installed
in the current environment. Filter the matrix on the command line:

```bash
pytest --nw-backends="pandas,polars[lazy]"
pytest --all-nw-backends
```

## Type aliases

::: narwhals.testing.typing
    handler: python
    options:
      show_root_heading: false
      heading_level: 3
      members:
        - Data
        - FrameConstructor
        - EagerFrameConstructor
        - LazyFrameConstructor
